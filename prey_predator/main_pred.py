import os

from cv2 import cv2
import torchvision.transforms as T

import robobo
from foraging.main_fora import save_training_results
from models.cnn_classifier import CNN
from models.mlp_classifier import MLP
from prey_predator import Prey
from utils import retrieve_network, get_ir_signal, save_network, classifier_network_testing, train_classifier_network, \
    prep_images_datasets, prepare_mlp_datasets
import torch
import datasets.generate_prey_movement_dataset as prey_movement_dataset
import datasets.generate_prey_images_dataset as prey_image_dataset
from datasets.generate_prey_images_dataset import ACTIONS as images_actions


device = "cpu"
# device = "cuda" if torch.cuda.is_available() is True else "cpu"
models_folder = "./models/"
datasets_folder = "./datasets/pickled_datasets/"
images_folder = "./images/"

predator_movement_model = models_folder + "movement_network.pt"
prey_movement_model = models_folder + "prey_movement_network.pt"
images_model = models_folder + "images_network.pt"





def main(
        data_package="data48.pkl",
        labels_package="labels48.pkl",
        batches: int=16,
        validation: float=0.15,
        testing: float=0.15,
        epochs: int=10,
        mlp_hidden_nodes: int=100,
        learning_rate: float=0.001):
    train_cnn = False
    train_prey_mlp = True
    train_rnn = False
    execute = False

    if train_cnn is True:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(0, 1)
        ])

        cnn = retrieve_network(Model=CNN,
                               device=device,
                               output_nodes=4,
                               network_name=images_model)

        train_dataset, validation_dataset, testing_dataset = prep_images_datasets(
            datasets_folder=datasets_folder,
            image_dataset=prey_image_dataset,
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

    if train_prey_mlp is True:
        mlp = retrieve_network(input_nodes=8,
                               output_nodes=6,
                               hidden_nodes=mlp_hidden_nodes,
                               Model=MLP,
                               device=device,
                               network_name=prey_movement_model)

        print("Generating MLP dataset")
        mlp_train_loader, mlp_validation_loader, mlp_test_loader = prepare_mlp_datasets(prey_movement_dataset, batches,
                                                                                        device)
        training_points = len(mlp_train_loader) * batches
        testing_points = len(mlp_test_loader) * batches
        validation_points = len(mlp_validation_loader) * batches

        print("Training MLP")
        mlp, training_loss, validation_loss = train_classifier_network(network=mlp,
                                                                       train_dataset=mlp_train_loader,
                                                                       epochs=epochs,
                                                                       device=device,
                                                                       validation_dataset=mlp_validation_loader,
                                                                       learning_rate=learning_rate)
        del mlp_train_loader
        del mlp_validation_loader

        print("Testing MLP")
        accuracy, _, _ = classifier_network_testing(network=mlp,
                                                    test_dataset=mlp_test_loader,
                                                    batches=batches,
                                                    device=device)
        print("MLP accuracy: " + str(accuracy))
        del mlp_test_loader

        print("Saving MLP")
        save_network(mlp, prey_movement_model)
        print("Saving MLP training results")
        save_training_results(training_loss=training_loss,
                              validation_loss=validation_loss,
                              testing_accuracy=accuracy,
                              model="PREY_MLP",
                              batches=batches,
                              epochs=epochs,
                              learning_rate=learning_rate,
                              training_points=training_points,
                              testing_points=testing_points,
                              validation_points=validation_points)

    if train_rnn is True:
        pass

    if execute is True:
        rob = robobo.SimulationRobobo().connect(address="127.0.0.1", port=19997)
        rob.play_simulation()
        prey_controller = robobo.SimulationRoboboPrey(number="#0")
        prey_controller.connect()
        prey = Prey(prey_controller, level=2)
        prey.start()

        nn = retrieve_network(input_nodes=5, hidden_nodes=100, output_nodes=6, Model=MLP, device=device, network_name=predator_movement_model)
        retrieve_images(nn, rob)

        rob.pause_simulation()
        # Stopping the simualtion resets the environment
        rob.stop_world()
