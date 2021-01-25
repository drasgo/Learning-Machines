import os

from cv2 import cv2
import torchvision.transforms as T

import robobo
from foraging.main_fora import save_training_results
from models.cnn_classifier import CNN
from models.mlp_classifier import MLP
from prey_predator import Prey
from utils import retrieve_network, get_ir_signal, save_network, classifier_network_testing, train_classifier_network, \
    prep_images_datasets
import torch
import datasets.generate_movement_dataset as dataset
from datasets import generate_prey_images_dataset


device = "cpu"
# device = "cuda" if torch.cuda.is_available() is True else "cpu"
models_folder = "./models/"
datasets_folder = "./datasets/pickled_datasets/"
images_folder = "./images/"

mlp_movement_model = models_folder + "movement_network.pt"
images_model = models_folder + "images_network.pt"


def check_folder_existence(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        pass


def retrieve_images(nn, rob):
    counter = 0
    prev_out = -1
    index = 0
    check_folder_existence("images/")
    for _, _, images in os.walk("images/"):
        index = len(images)

    while True:
        if rob.is_simulation_running() is False:
            print("Error with simulation")
            break

        index += 1
        image = rob.get_image_front()
        cv2.imwrite("images/prey_" + str(index) + ".png", image)

        # IR reading
        ir = get_ir_signal(rob, device)
        print("ROB Irs: {}".format(ir))
        # Net output
        outputs = nn(ir)
        _, output = torch.max(outputs.data, 1)
        output = output.item()
        # Check if it got stuck

        if output == prev_out and output != 0:
            counter += 1
            if counter >= 3:
                output = 5

        # Motors actuators
        print(output)
        left_motor = dataset.ACTIONS[output]["motors"][0]
        right_motor = dataset.ACTIONS[output]["motors"][1]
        time = dataset.ACTIONS[output]["time"]

        if prev_out != output or (prev_out == output and output != 0):
            rob.move(left_motor, right_motor, time)

        prev_out = output


def main(
        data_package="data48.pkl",
        labels_package="labels48.pkl",
        batches=16,
        validation=0.15,
        testing=0.15,
        epochs=10,
        learning_rate=0.001):
    train_cnn = True
    train_rnn = False
    execute = False

    if train_cnn is True:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(0, 1)
        ])

        cnn = retrieve_network(Model=CNN,
                               device=device,
                               output_nodes=6,
                               network_name=images_model)

        train_dataset, validation_dataset, testing_dataset = prep_images_datasets(
            datasets_folder=datasets_folder,
            image_dataset=generate_prey_images_dataset,
            images_folder=images_folder,
            data_package=data_package,
            labels_package=labels_package,
            batch_size=batches,
            validation_percentage=validation,
            testing_percentage=testing,
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
        save_network(cnn, images_model)
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


    if execute is True:
        rob = robobo.SimulationRobobo().connect(address="127.0.0.1", port=19997)
        rob.play_simulation()
        prey_controller = robobo.SimulationRoboboPrey(number="#0")
        prey_controller.connect()
        prey = Prey(prey_controller, level=2)
        prey.start()

        nn = retrieve_network(input_nodes=5, hidden_nodes=100, output_nodes=6, Model=MLP, device=device, network_name=mlp_movement_model)
        retrieve_images(nn, rob)

        rob.pause_simulation()
        # Stopping the simualtion resets the environment
        rob.stop_world()
