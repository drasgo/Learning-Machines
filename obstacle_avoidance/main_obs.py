import robobo
from pynput import keyboard

import datasets.generate_movement_dataset as dataset
from models.mlp_classifier import MLP
import torch

from utils import retrieve_network, save_network, prepare_mlp_datasets, get_ir_signal, train_classifier_network, \
    classifier_network_testing

PRESSED = False
device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batches = 10
mlp_model = "/models/movement_model.pt"


def keyboard_action(key):
    global PRESSED
    if key == keyboard.Key.enter:
        PRESSED = True


def main():
    train = False

    if train is True:
        nn = retrieve_network(input_nodes=5, hidden_nodes=100, output_nodes=6, Model=MLP, device=device)
        train_loader, test_loader = prepare_mlp_datasets(dataset, batches, device)

        nn, trainig_loss, validation_loss = train_classifier_network(nn, train_loader, 10, device)
        accuracy, _, _ = classifier_network_testing(nn, test_loader, batches, device)

        save_network(nn)
        print("network saved")
        print(accuracy)

    else:
        l1 = keyboard.Listener(on_press=lambda key: keyboard_action(key))
        l1.start()
        counter = 0
        prev_out = -1
        nn = retrieve_network(input_nodes=5, output_nodes=6, Model=MLP, device=device, network_name=mlp_model)
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

            print(output)

            # Motors actuators
            left_motor = dataset.ACTIONS[output]["motors"][0]
            right_motor = dataset.ACTIONS[output]["motors"][1]
            time = dataset.ACTIONS[output]["time"]
            print("l " + str(left_motor) + ", r: " + str(right_motor))

            if prev_out != output or (prev_out == output and output != 0):
                rob.move(left_motor, right_motor, time)

            prev_out = output

        rob.pause_simulation()
        # Stopping the simualtion resets the environment
        rob.stop_world()
