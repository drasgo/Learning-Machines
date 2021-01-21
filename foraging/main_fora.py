import json
import os
import pickle
import random
import torch
import torchvision.transforms as T
from pynput import keyboard

import datasets.generate_images_dataset as image_dataset
import datasets.generate_movement_dataset as movement_dataset
import robobo
from models.cnn_classifier import CNN
from models.mlp_classifier import MLP
from utils import retrieve_network, save_network, prepare_mlp_datasets, get_ir_signal, train_classifier_network, \
    classifier_network_testing, prepare_dataset

PRESSED = False
device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_folder = "./models/"
datasets_folder = "./datasets/"
images_folder = "./images/"
movement_model = models_folder + "movement_network.pt"
images_model = models_folder + "images_network.pt"

def keyboard_action(key):
    global PRESSED
    if key == keyboard.Key.enter:
        PRESSED = True


def prep_dataset_v2(data_package, labels_package, batch_size, validation, testing, transform):
    data_package_path = datasets_folder + "/" + data_package
    labels_package_path = datasets_folder + "/" + labels_package
    if not os.path.exists(data_package_path) or not os.path.exists(labels_package_path):
        _, _ = image_dataset.generate_dataset(images_folder)

    with open(labels_package_path, "rb") as fp:
        labels = pickle.load(fp)
        assert isinstance(labels, list)
    print("loaded labels")

    with open(data_package_path, "rb") as fp:
        data = pickle.load(fp)
        assert isinstance(data, list)
    print("loaded data")

    train_data = data[int((validation + testing)* len(data)):]
    train_labels = labels[int((validation + testing) * len(labels)):]
    print("divided train data")

    validation_data = data[:int(validation * len(data))]
    validation_labels = labels[: int(validation * len(data))]
    print("divided validation data")

    testing_data = data[int(validation*len(data)): int(validation*len(data)) + int(testing * len(data))]
    testing_labels = labels[int(validation*len(labels)): int(validation*len(labels)) + int(testing * len(labels))]
    print("divided test data")

    del labels
    del data
    print("removed data and labels from memory")

    train_dataset = prepare_dataset(train_data, train_labels, batch_size, transform)
    print("created train dataset, of " + str(len(train_data)) + " inputs")
    del train_data
    del train_labels
    print("removed train data and train labels from memory")

    if len(validation_data) > 0:
        validation_dataset = prepare_dataset(validation_data, validation_labels, batch_size, transform)
        print("created validation dataset, of " + str(len(validation_data)) + " inputs")
        del validation_data
        del validation_labels
        print("removed validation data and validation labels from memory")
    else:
        validation_dataset = None

    if len(testing_data) > 0:
        test_dataset = prepare_dataset(testing_data, testing_labels, batch_size, transform)
        print("created testing dataset, of " + str(len(testing_data)) + " inputs")
        del testing_data
        del testing_labels
        print("removed test data and test labels from memory")
    else:
        test_dataset = None

    return train_dataset, validation_dataset, test_dataset


def save_training_results(
        training_loss: list,
        validation_loss: list,
        testing_accuracy: float,
        model,
        batches: int,
        epochs: int,
        learning_rate: float,
        training_points: int,
        testing_points: int,
        validation_points: int
):
    if os.path.exists("result.json"):
        with open("result.json", "r") as fp:
            result = json.load(fp)
    else:
        result = {}
    index = len(result)
    result[index] = {
        "type": model,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "testing_accuracy": testing_accuracy,
        "batches": batches,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "training_points": training_points,
        "testing_points": testing_points,
        "validation_points": validation_points
    }

    with open("result.json", "w") as fp:
        json.dump(result, fp)


def labelling_testing(image):
    im_list = image.tolist()
    # im_list = im.tolist()

    horizontal_position = -1
    # Iterate through rows (from the bottom to  the topof the image,  it should recognize closer targets)
    for row_index in range(len(im_list) - 1, -1, -1):

        # Iterate through columns (meaning that each "column index" is the index of a pixel)
        for column_index in range(len(im_list[row_index])):
            # Pixel cell with r g b values
            pixel = im_list[row_index][column_index]
            # If the pixel is green look in the same row if there are other green pixels. If so, check what is the last
            # green pixel and save it. So, the average green pixel position on the horizontal axis is the average of these two
            if (pixel[0] < 165 and pixel[1] > 195 and pixel[2] < 175) or (
                    pixel[0] < 20 and pixel[1] > 150 and pixel[2] < 20):
                last_pixel = -1

                for other_pixels in range(column_index + 1, len(im_list[row_index])):
                    other_pixel = im_list[row_index][other_pixels]
                    if (other_pixel[0] < 165 and other_pixel[1] > 205 and other_pixel[2] < 175) or \
                            (other_pixel[0] < 20 and other_pixel[1] > 150 and other_pixel[2] < 20):
                        continue
                    last_pixel = other_pixels - 1
                    break

                if last_pixel == -1:
                    average_pixel = column_index

                else:
                    average_pixel = (last_pixel + column_index) / 2

                # the horizontal position is a perccentage between 0 (far left) and 1 (far right). This will be
                # divided in 5 slots which will be used for finding the labels of the 5 images.
                # If no green pixel was found, than the horiizontal_axis is -1 and the label will be "no target in the image"
                horizontal_position = average_pixel / len(im_list[row_index])
                break
        if horizontal_position != -1:
            break

    #     choose label for moving towards the block (which is in the picture)
    return image_dataset.generate_label(horizontal_position)


def main(
        data_package: str,
        labels_package: str,
        epochs: int=10,
        batches: int=10,
        mlp_hidden_nodes: int=100,
        validation: float=0.15,
        testing: float=0.15,
        learning_rate: float=0.01
):
    train = True

    if train is True:
        if not os.path.exists(movement_model):
            device_mlp = "cpu"
            print("Retrieving MLP model")
            mlp = retrieve_network(input_nodes=5,
                                   output_nodes=6,
                                   hidden_nodes=mlp_hidden_nodes,
                                   Model=MLP,
                                   device=device_mlp,
                                   network_name=movement_model)

            print("Generating MLP dataset")
            mlp_train_loader, mlp_validation_loader, mlp_test_loader = prepare_mlp_datasets(movement_dataset, batches, device_mlp)
            training_points = len(mlp_train_loader) * batches
            testing_points = len(mlp_test_loader) * batches
            validation_points = len(mlp_validation_loader) * batches

            print("Training MLP")
            mlp, training_loss, validation_loss = train_classifier_network(network=mlp,
                                                                         train_dataset=mlp_train_loader,
                                                                         epochs=epochs,
                                                                         device=device_mlp,
                                                                         validation_dataset=mlp_validation_loader,
                                                                         learning_rate=learning_rate)
            del mlp_train_loader
            del mlp_validation_loader

            print("Testing MLP")
            accuracy, _, _ = classifier_network_testing(network=mlp,
                                                        test_dataset=mlp_test_loader,
                                                        batches=batches,
                                                        device=device_mlp)
            print("MLP accuracy: " + str(accuracy))
            del mlp_test_loader

            print("Saving MLP")
            save_network(mlp, movement_model)
            print("Saving MLP training results")
            save_training_results(training_loss=training_loss,
                                  validation_loss=validation_loss,
                                  testing_accuracy=accuracy,
                                  model="MLP",
                                  batches=batches,
                                  epochs=epochs,
                                  learning_rate=learning_rate,
                                  training_points=training_points,
                                  testing_points=testing_points,
                                  validation_points=validation_points)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(0, 1)
        ])
        for batches in [16, 10]:
            for learning_rate in [0.01, 0.001, 0.1]:
                for data_file, labels_file, network_name in [["data15.pkl", "labels15.pkl", "15"],
                                                             ["data25.pkl", "labels25.pkl", "25"],
                                                             ["data35.pkl", "labels35.pkl", "35"],
                                                             ["data45.pkl", "labels45.pkl", "45"]]:
                    model = models_folder + "images_points" + network_name + "_lr" + str(learning_rate) + " _bat" + str(batches) + "_network.pt"
                    print("Retrieving CNN model")
                    cnn = retrieve_network(Model=CNN,
                                           device=device,
                                           output_nodes=6,
                                           network_name=model)

                    print("creating dataset")
                    train_dataset, validation_dataset, testing_dataset = prep_dataset_v2(data_package=data_file,
                                                                                         labels_package=labels_file,
                                                                                         batch_size=batches,
                                                                                         validation=validation,
                                                                                         testing=testing,
                                                                                         transform=transform)
                    training_points = len(train_dataset) * batches
                    testing_points = len(testing_dataset) * batches
                    validation_points = len(validation_dataset) * batches
                    print("Training CNN")
                    cnn, train_loss, validation_loss = train_classifier_network(network=cnn,
                                                                                train_dataset=train_dataset,
                                                                                epochs=epochs,
                                                                                device=device,
                                                                                validation_dataset=validation_dataset,
                                                                                learning_rate=learning_rate)
                    del train_dataset
                    del validation_dataset
                    print("Removed train dataset and validation dataset")
                    accuracy, _, _ = classifier_network_testing(cnn, testing_dataset, batches, device)
                    print("CNN accuracy:" + str(accuracy))
                    del testing_dataset
                    print("Removed test dataset")
                    model = models_folder + "images_points" + network_name + "_lr" + str(learning_rate) + " _bat" + str(batches) + "_network.pt"

                    print("Saving CNN")
                    save_network(cnn, model)
                    print("Saving CNN training results")
                    save_training_results(training_loss=train_loss,
                                          validation_loss=validation_loss,
                                          testing_accuracy=accuracy,
                                          model="CNN",
                                          batches=batches,
                                          epochs=epochs,
                                          learning_rate=learning_rate,
                                          training_points=training_points,
                                          testing_points=testing_points,
                                          validation_points=validation_points)


    l1 = keyboard.Listener(on_press=lambda key: keyboard_action(key))
    l1.start()

    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
    rob.set_phone_tilt(26, 100)

    mlp = retrieve_network(input_nodes=5, hidden_nodes=100, output_nodes=6, Model=MLP, device=device, network_name=movement_model)
    cnn = retrieve_network(Model=CNN, device=device, output_nodes=6, network_name=images_model)

    while PRESSED is False:
        if PRESSED is True:
            print("Finishing")
            break

        rob.play_simulation()
        rob.play_simulation()
        rob.set_phone_tilt(26, 100)

        counter = 0
        prev_out = -1
        while rob.collected_food() < 6:
            if PRESSED is True:
                break

            # IR reading
            ir = get_ir_signal(rob, device)
            im = rob.get_image_front()
            image = T.ToPILImage()(im)

            image = T.Compose([
                T.ToTensor(),
                T.Normalize(0, 1)
            ])(image).to(device)

            print("ROB Irs: {}".format(ir))
            # Net output
            mlp_output = mlp(ir)
            cnn_output = cnn(image[None, ...])

            _, mlp_output = torch.max(mlp_output.data, 1)
            _, cnn_output = torch.max(cnn_output.data, 1)

            mlp_output = mlp_output.item()
            cnn_output = cnn_output.item()

            # Check if it got stuck
            if (cnn_output == 5 and mlp_output != 0) and \
                    ((cnn_output in [0, 1, 2] and mlp_output in [3, 4]) or
                     (cnn_output in [0, 3, 4] and mlp_output in [1, 2])):
                print("in mlp")
                output = mlp_output
                dataset = movement_dataset

            else:
                print("in cnn")
                output = cnn_output
                # output = labelling_testing(im)
                dataset = image_dataset
                if output == 5:
                    counter += 1
                    if counter == 3:
                        output = mlp_output
                        dataset = movement_dataset
                else:
                    counter = 0

            print(output)
            # Motors actuators
            left_motor = dataset.ACTIONS[output]["motors"][0]
            right_motor = dataset.ACTIONS[output]["motors"][1]

            time = dataset.ACTIONS[output]["time"]

            if prev_out != output or output != 0:
                rob.move(left_motor, right_motor, time)

            prev_out = output

        rob.pause_simulation()
        print(rob.collected_food())
        # Stopping the simualtion resets the environment
        rob.stop_world()
