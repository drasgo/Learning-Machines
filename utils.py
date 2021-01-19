import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.Dataset import Dataset


def save_network(network, network_name: str="network.pt"):
    torch.save(network.state_dict(), network_name)


def retrieve_network(Model, device, input_nodes: int=None, output_nodes: int=None, network_name: str="network.pt"):
    if input_nodes is not None and output_nodes is not None:
        model = Model(input_nodes, output_nodes).to(device)
    else:
        model = Model(output_nodes).to(device)
    if os.path.exists(network_name):
        print("ciao")
        model.load_state_dict(torch.load(network_name))
        model.eval()
    return model


def prepare_dataset(train_data, train_pred, batch_size, transform):
    trainset = Dataset(train_data, train_pred, transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader

def prepare_datasets(dataset, batch_size: int, device, train_size: int = 500000, test_size: int=50000, transform=None):
    train_data, train_pred = dataset.generate_dataset(train_size)
    test_data, test_pred = dataset.generate_dataset(test_size)

    if transform is None:
        train_data = torch.tensor(train_data, device=device)
        train_pred = torch.tensor(train_pred, device=device)

        test_data = torch.tensor(test_data, device=device)
        test_pred = torch.tensor(test_pred, device=device)

    trainloader = prepare_dataset(train_data, train_pred, batch_size, transform)
    testloader = prepare_dataset(test_data, test_pred, batch_size, transform)

    return trainloader, testloader


def get_ir_signal(rob, device):
    ir_signal = np.log(np.array(rob.read_irs())) / 10
    ir_signal[ir_signal == np.NINF] = 0
    ir_signal = torch.Tensor(ir_signal[-5:]).to(device)
    ir_signal = ir_signal.view(size=(1, ir_signal.size(0)))
    return ir_signal


def validation(network, criterion, validation_dataset, device):
    epoch_validation_loss = []
    for batch_idx, (inputs, labels) in tqdm(enumerate(validation_dataset)):
        outs = network(inputs.to(device))
        loss = criterion(outs, labels.to(device))
        loss.backward()
        epoch_validation_loss.append(loss.item())
    return epoch_validation_loss


def train_classifier_network(network, train_dataset, epochs, device, validation_dataset=None, learning_rate=0.01):
    total_iterations = 0
    train_losses = []
    validation_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_train_loss = []

        for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataset)):
            optimizer.zero_grad()
            outs = network(inputs.to(device))
            loss = criterion(outs, labels.to(device))
            loss.backward()
            optimizer.step()

            total_iterations += 1
            epoch_train_loss.append(loss.item())
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 2000)
                )
                running_loss = 0.0

        train_losses.append(epoch_train_loss)
        if validation_dataset is not None:
            validation_loss = validation(network, criterion, validation_dataset, device)
            validation_losses.append(validation_loss)
    return network, train_losses, validation_losses


def classifier_network_testing(network, test_dataset, batches, device):
    corr = 0
    tot = 0
    counter = 0
    with torch.no_grad():
        for data, labels in test_dataset:
            counter += 1
            outs = network(data.to(device))
            _, predicted = torch.max(outs.data, 1)
            tot += labels.size(0)
            corr += (predicted == labels.to(device)).sum().item()
    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test data: %d %%" % (counter * batches, acc))
    return acc, corr, tot
