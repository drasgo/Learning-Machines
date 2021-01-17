import robobo
from pynput import keyboard
from tqdm import tqdm

import datasets.generate_movement_dataset as dataset
from models.cnn_classifier import CNN
import torch

from utils import retrieve_network, save_network, prepare_datasets, get_ir_signal


PRESSED = False
device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batches = 10


def train_network(network, train_dataset, epochs:int, learning_rate: int=0.01):
    pass


def network_testing(network, test_dataset):
    pass


def keyboard_action(key):
    global PRESSED
    if key == keyboard.Key.enter:
        PRESSED = True


def main():
    train = True

    if train is True:
        nn = retrieve_network(5, 6, CNN, device)
        train_loader, test_loader = prepare_datasets(dataset, batches, device)
        nn, _ = train_network(nn, train_loader, 10)
        accuracy, _, _ = network_testing(nn, test_loader)
        save_network(nn)
        print("network saved")
        print(accuracy)

    else:
        l1 = keyboard.Listener(on_press=lambda key: keyboard_action(key))
        l1.start()
        counter = 0
        prev_out = -1
        nn = retrieve_network(5, 6, CNN, device)
        rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        while PRESSED is False:
            if PRESSED is True or rob.is_simulation_running() is False:
                print("Error with simulation")
                break

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
            left_motor = dataset.ACTIONS[output]["motors"][0]
            right_motor = dataset.ACTIONS[output]["motors"][1]

            if output != 0:
                time = dataset.ACTIONS[output]["time"]
            else:
                time = None

            if prev_out != output or (prev_out == output and output != 0):
                rob.move(left_motor, right_motor, time)

            prev_out = output

        rob.pause_simulation()
        # Stopping the simualtion resets the environment
        rob.stop_world()