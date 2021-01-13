import robobo
from pynput import keyboard
from tqdm import tqdm

import obstacle_avoidance.generate_dataset as dataset
from obstacle_avoidance.model import Model,Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np


PRESSED = False


def train_network(network, train_input, epochs, learning_rate=0.01):
    total_iterations = 0
    total_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_input)):

            optimizer.zero_grad()
            outs = network(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            total_iterations += 1
            total_losses.append(loss.item())
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 2000)
                )
                running_loss = 0.0
    return network, total_losses


def test_network(network, test_load):
    corr = 0
    tot = 0
    counter = 0
    with torch.no_grad():
        for data, labels in test_load:
            counter += 1
            outs = network(data)
            _, predicted = torch.max(outs.data, 1)
            tot += labels.size(0)
            corr += (predicted == labels).sum().item()
    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test data: %d %%" % (counter * batches, acc))
    return acc, corr, tot


def save_network(network):
    torch.save(network.state_dict(), "network.pt")


def retrieve_network():
    model = Model(5, 6)
    model.load_state_dict(torch.load("network.pt"))
    model.eval()
    return model


def prepare_datasets():
    train_data, train_pred = dataset.generate_dataset(500000)
    # print(train_data[:15])
    # print(train_pred[:15])
    # input()
    test_data, test_pred = dataset.generate_dataset(50000)

    train_data = torch.Tensor(train_data)
    train_pred = torch.Tensor(train_pred)
    trainset = Dataset(train_data, train_pred)
    trainloader = DataLoader(trainset, batch_size=batches, shuffle=True)

    test_data = torch.Tensor(test_data)
    test_pred = torch.Tensor(test_pred)
    testset = Dataset(test_data, test_pred)
    testloader = DataLoader(testset,batch_size=batches, shuffle=True)
    return trainloader, testloader


def get_input():
    ir_signal = np.log(np.array(rob.read_irs())) / 10
    ir_signal[ir_signal == np.NINF] = 0
    ir_signal = torch.Tensor(ir_signal[-5:])
    ir_signal = ir_signal.view(size=(1, ir_signal.size(0)))
    return ir_signal

def keyboard_action(key):
    global PRESSED
    if key == keyboard.Key.enter:
        PRESSED = True


if __name__ == "__main__":
    train = True
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    if train is True:
        nn = Model(input_size=5, output_size=6)
        batches = 10
        train_loader, test_loader = prepare_datasets()
        nn, _ = train_network(nn, train_loader, 10)
        accuracy, _, _ = test_network(nn, test_loader)
        save_network(nn)
        print("network saved")
        print(accuracy)

    else:
        l1 = keyboard.Listener(on_press=lambda key: keyboard_action(key))
        l1.start()
        counter = 0
        prev_out = -1
        nn = retrieve_network()
        rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
        while PRESSED is False:
            if PRESSED is True or rob.is_simulation_running() is False:
                print("Error with simulation")
                break
            # IR reading
            ir = get_input()
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
            print("l " + str(left_motor) + ", r: " + str(right_motor))

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
