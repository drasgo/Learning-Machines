import json
import os
import random

import PIL
import cv2
import torchvision.transforms as T
from tqdm import tqdm

import robobo
import numpy as np
from pynput import keyboard

import datasets.generate_movement_dataset as movement_dataset
import datasets.generate_images_dataset as image_dataset
from models.cnn_classifier import CNN
import torch

from models.mlp_classifier import MLP
from utils import retrieve_network, save_network, prepare_datasets, get_ir_signal, train_classifier_network, \
    classifier_network_testing, prepare_dataset

PRESSED = False
# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batches = 10
models_folder = "./models/"
movement_model = models_folder + "movement_network.pt"
images_model = models_folder + "images_network.pt"

def keyboard_action(key):
    global PRESSED
    if key == keyboard.Key.enter:
        PRESSED = True


def prep_dataset(filename, transform, validation_perc=0.0, batch_size=5):
    with open(filename, "r") as fp:
        chunk = json.load(fp)
    validation_dataset= None
    train_data = np.array(chunk["images"][validation_perc*len(chunk["images"]) :], dtype=np.uint8)
    train_labels = np.array(chunk["labels"][validation_perc*len(chunk["images"]) :], dtype=np.uint8)
    train_dataset = prepare_dataset(train_data, train_labels, batch_size, transform)

    if validation_perc != 0.0:
        validation_data = np.array(chunk["images"][: validation_perc*len(chunk["images"])], dtype=np.uint8)
        validation_labels = np.array(chunk["labels"][: validation_perc*len(chunk["images"])], dtype=np.uint8)

        validation_dataset = prepare_dataset(validation_data, validation_labels, batch_size, transform)
    return train_dataset, validation_dataset


def main():
    train = True
    counter = 0
    prev_out = -1

    if train is True:

        transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(0, 1)
        ])

        if not os.path.exists(movement_model):
            device_mlp = "cpu"
            mlp = retrieve_network(input_nodes=5, output_nodes=6, Model=MLP, device=device_mlp,
                                   network_name=movement_model)
            print("Generating MLP dataset")
            mlp_train_loader, mlp_test_loader = prepare_datasets(movement_dataset, batches, device_mlp)
            print("Training MLP")
            mlp, loss = train_classifier_network(mlp, mlp_train_loader, 10, device_mlp)
            print("Testing MLP")
            accuracy, _, _ = classifier_network_testing(mlp, mlp_test_loader, batches, device_mlp)
            print("MLP accuracy: " + str(accuracy))
            print("Saving MLP")
            save_network(mlp, movement_model)
            return

        _,_,files = next(os.walk("datasets/images/"))
        for file_index in range(len(files)):
            cnn = retrieve_network(CNN, device, output_nodes=6, network_name="movement_network.pt")
            print("Generating CNN dataset")
            val_perc = 0

            if file_index != len(files) - 1:
                if file_index > len(files)/2:
                    val_perc = 0.15

                cnn_train_loader, validation_loader = prep_dataset("datasets/images/" + files[file_index],
                                                                   batch_size=5,
                                                                   transform=transform,
                                                                   validation_perc=val_perc)

                print("Training CNN")
                cnn, train_loss, validation_loss = train_classifier_network(cnn, cnn_train_loader, 10, device)

            else:
                print("Testing CNN")
                cnn_train_loader, _ = prep_dataset("datasets/images/" + files[file_index],
                                                                   batch_size=5,
                                                                   transform=transform)

                accuracy, _, _ = classifier_network_testing(cnn, cnn_train_loader, batches, device)
                print("CNN accuracy:" + str(accuracy))

            del cnn_train_loader
            print("Saving CNN")
            save_network(cnn, images_model)


    else:
        l1 = keyboard.Listener(on_press=lambda key: keyboard_action(key))
        l1.start()

        rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        rob.play_simulation()
        rob.set_phone_tilt(26, 100)

        mlp = retrieve_network(input_nodes=5, output_nodes=6, Model=MLP, device=device, network_name=movement_model)
        cnn = retrieve_network(Model=CNN, device=device, output_nodes=6, network_name=images_model)

        while PRESSED is False:
            if PRESSED is True or rob.is_simulation_running() is False:
                print("Finishing")
                break

            # IR reading
            ir = get_ir_signal(rob, device)
            image = rob.get_image_front()
            image = T.ToPILImage()(image)
            print(image.size)
            image.show()

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
            if mlp_output != 0 and \
                    ((cnn_output in [0, 1, 2] and mlp_output in [3, 4]) or (
                            cnn_output in [0, 3, 4] and mlp_output in [1, 2])):
                print("in mlp")
                print(mlp_output)
                output = mlp_output
                dataset = movement_dataset

            else:
                print("in cnn")
                print(cnn_output)
                output = cnn_output
                dataset = image_dataset
                if output == 5:
                    output = random.choice([2, 4])

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

# def main_temp():
#     # temp()
#     check_last_pulled_folder()
#     high = check_highest()
#     nn = retrieve_network(input_nodes=5, output_nodes=6, Model=MLP, device=device, network_name=movement_model)
#     rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
#     while high < 200000:
#         rob.play_simulation()
#         rob.play_simulation()
#         rob.set_phone_tilt(26, 100)
#
#         counter = 0
#         prev_out = -1
#         while rob.collected_food() < 4:
#             # IR reading
#             ir = get_ir_signal(rob, device)
#             high += 1
#             image = rob.get_image_front()
#             cv2.imwrite("images/image_"+str(high)+".png", image)
#             # Net output
#             outputs = nn(ir)
#             _, output = torch.max(outputs.data, 1)
#             output = output.item()
#             # Check if it got stuck
#
#             if output == prev_out and output != 0:
#                 counter += 1
#                 if counter >= 3:
#                     output = 5
#
#             # Motors actuators
#             left_motor = movement_dataset.ACTIONS[output]["motors"][0]
#             right_motor = movement_dataset.ACTIONS[output]["motors"][1]
#             time = movement_dataset.ACTIONS[output]["time"]
#
#             if prev_out != output or (prev_out == output and output != 0):
#                 rob.move(left_motor, right_motor, time)
#
#             prev_out = output
#
#         rob.pause_simulation()
#         # Stopping the simualtion resets the environment
#         rob.stop_world()
