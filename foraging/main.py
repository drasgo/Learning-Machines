import torchvision.transforms as T
import robobo
from pynput import keyboard

import datasets.generate_movement_dataset as movement_dataset
import datasets.generate_images_dataset as image_dataset
from models.cnn_classifier import CNN
import torch

from models.mlp_classifier import MLP
from utils import retrieve_network, save_network, prepare_datasets, get_ir_signal, train_classifier_network, \
    classifier_network_testing

PRESSED = False
device = "cpu"
batches = 10


def keyboard_action(key):
    global PRESSED
    if key == keyboard.Key.enter:
        PRESSED = True


def main():
    train = True

    # rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
    # rob.set_phone_tilt(26, 100)
    # rob.move(50, 50, 1000)
    # p = rob.read_irs()
    # print(p)
    # input()
    counter = 0
    prev_out = -1

    # im = PIL.Image.open("ball.png")
    # im2 = PIL.Image.open("landscape.png")
    #
    # image_height = im2.size[1]
    # image_width = im2.size[0]
    #
    # offset = random.randint(-int(image_height/20), int(image_height/3)) - im.size[1]/2
    # ball_x = random.randint(0, image_width)
    # ball_y = int((2*image_height / 3) + offset)
    #
    # im3 = im2.copy()
    # im3.paste(im, (ball_x, ball_y))
    # im3.show()
    #
    # input()
    # pass

    if train is True:
        cnn = retrieve_network(CNN, device, network_name="movement_network.pt")
        mlp = retrieve_network(input_nodes=5, output_nodes=6, Model=MLP, device=device, network_name="images_network.pt")

        transform = T.Compose([
            T.RandomRotation((-7, 7)),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        cnn_train_loader, cnn_test_loader = prepare_datasets(image_dataset, batches, device, train_size=50000, transform=transform)
        print(cnn_train_loader)
        input()
        mlp_train_loader, mlp_test_loader = prepare_datasets(movement_dataset, batches, device)

        mlp, _ = train_classifier_network(mlp, mlp_train_loader, 10, device)
        accuracy, _, _ = classifier_network_testing(mlp, mlp_test_loader, batches)

        cnn, _ = train_classifier_network(cnn, cnn_train_loader, 10, device)
        accuracy, _, _ = classifier_network_testing(cnn, cnn_test_loader, batches)

        save_network(mlp, "movement_network.pt")
        save_network(cnn, "images_network.pt")

        print("networks saved")
        print(accuracy)

    else:
        l1 = keyboard.Listener(on_press=lambda key: keyboard_action(key))
        l1.start()

        nn = retrieve_network(5, 6, CNN, device)
        rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        # rob.set_phone_tilt(26, 100)
        # image = rob.get_image_front()

        while PRESSED is False:
            if PRESSED is True or rob.is_simulation_running() is False:
                print("Finishing")
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
            left_motor = image_dataset.ACTIONS[output]["motors"][0]
            right_motor = image_dataset.ACTIONS[output]["motors"][1]

            if output != 0:
                time = image_dataset.ACTIONS[output]["time"]
            else:
                time = None

            if prev_out != output or (prev_out == output and output != 0):
                rob.move(left_motor, right_motor, time)

            prev_out = output

        rob.pause_simulation()
        # Stopping the simualtion resets the environment
        rob.stop_world()