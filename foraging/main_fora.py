import json
import os
import pickle
import time

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


def execution(rob, cnn, mlp, counter, prev_out, prev_model):
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
    if (cnn_output == 5 and mlp_output != 0) or \
        (cnn_output in [0, 1, 2] and mlp_output in [3, 4]) or \
        (cnn_output in [0, 3, 4] and mlp_output in [1, 2]) or \
            (cnn_output == 5 and counter > 4):
        print("in mlp")
        model = "mlp"
        output = mlp_output
        dataset = movement_dataset
        if prev_model == "mlp" and (prev_out == 4 or prev_out == 2):
            output = 5

    else:
        print("in cnn")
        model = "cnn"
        output = cnn_output
        dataset = image_dataset
        motors = dataset.ACTIONS[output]["motors"]

        if output == 5:
            counter += 1
            print(counter)
            if prev_model == "mlp" and \
                (prev_out in [1, 2] and motors[0] == max(motors)) or \
                (prev_out in [3, 4] and motors[1] == max(motors)):
                dataset.ACTIONS[output]["motors"] = (motors[1], motors[0])

        else:
            counter = 0

    print(output)
    # Motors actuators
    left_motor = dataset.ACTIONS[output]["motors"][0]
    right_motor = dataset.ACTIONS[output]["motors"][1]

    action_time = dataset.ACTIONS[output]["time"]

    if prev_out != output or output != 0:
        rob.move(left_motor, right_motor, action_time)

    return counter, output, model


def count_food(delta, max_time, rob, total):
    if rob.collected_food() != total and delta < 180:
        return False
    else:
        return True


def timer_func(delta, max_time, rob, total):
    if delta < max_time and rob.collected_food() != total:
        return False
    else:
        return True


def main(
        data_package: str,
        labels_package: str,
        mlp_model: str=movement_model,
        cnn_model: str=images_model,
        epochs: int=10,
        batches: int=10,
        mlp_hidden_nodes: int=100,
        validation: float=0.15,
        testing: float=0.15,
        learning_rate: float=0.01
):
    train_mlp = False
    train_cnn = False

    if train_mlp is True:
        device_mlp = "cpu"
        print("Retrieving MLP model")
        mlp = retrieve_network(input_nodes=5,
                               output_nodes=6,
                               hidden_nodes=mlp_hidden_nodes,
                               Model=MLP,
                               device=device_mlp,
                               network_name=mlp_model)

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
        save_network(mlp, mlp_model)
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

    if train_cnn is True:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(0, 1)
            ])
            print("Retrieving CNN model")
            cnn = retrieve_network(Model=CNN,
                                   device=device,
                                   output_nodes=6,
                                   network_name=cnn_model)

            print("creating dataset")
            train_dataset, validation_dataset, testing_dataset = prep_dataset_v2(data_package=data_package,
                                                                                 labels_package=labels_package,
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
            # model = models_folder + "images_points" + cnn_model + "_lr" + str(learning_rate) + " _bat" + str(batches) + "_network.pt"

            print("Saving CNN")
            save_network(cnn, cnn_model)
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


    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
    rob.set_phone_tilt(26, 100)

    mlp = retrieve_network(input_nodes=5, hidden_nodes=100, output_nodes=6, Model=MLP, device=device, network_name=movement_model)
    cnn = retrieve_network(Model=CNN, device=device, output_nodes=6, network_name=images_model)
    result = {}
    for count in range(5):
        # for condition in [timer_func]:
        for condition in [count_food, timer_func]:
            print("Iteration n°: " + str(count))

            rob.play_simulation()
            time.sleep(1)
            rob.play_simulation()
            rob.play_simulation()
            rob.set_phone_tilt(26, 100)

            counter = 0
            prev_out = -1
            prev_model = "N/A"
            timer = time.time()
            delta = 0
            print("Starting " + str(condition.__name__))

            while condition(delta, 30, rob, 7) is False:
                counter, prev_out, prev_model = execution(rob, cnn, mlp, counter, prev_out, prev_model)
                delta = time.time() - timer

            print("Finished " + str(condition.__name__) + " by collecting " +
                  str(rob.collected_food()) + " foods in " + str(delta) + " seconds")

            result[str(count) + "_" + str(condition.__name__) + "_scene1"] = {
                "food": rob.collected_food(),
                "time": delta
            }

            # Stopping the simualtion resets the environment
            rob.pause_simulation()
            rob.stop_world()

            with open("execution_results_scene1.json", "w") as fp:
                json.dump(result, fp)
            print("results saved in execution_results.json")
