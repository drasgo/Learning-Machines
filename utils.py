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
        model = Model().to(device)
    if os.path.exists("network.pt"):
        model.load_state_dict(torch.load(network_name))
        model.eval()
    return model


def prepare_datasets(dataset, batch_size: int, device, train_size: int = 500000, test_size: int=50000, transform=None):
    train_data, train_pred = dataset.generate_dataset(train_size)
    test_data, test_pred = dataset.generate_dataset(test_size)

    if transform is None:
        train_data = torch.tensor(train_data, device=device)
        train_pred = torch.tensor(train_pred, device=device)

        test_data = torch.tensor(test_data, device=device)
        test_pred = torch.tensor(test_pred, device=device)

    trainset = Dataset(train_data, train_pred, transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = Dataset(test_data, test_pred, transform)
    testloader = DataLoader(testset,batch_size=batch_size, shuffle=True)

    return trainloader, testloader


def get_ir_signal(rob, device):
    ir_signal = np.log(np.array(rob.read_irs())) / 10
    ir_signal[ir_signal == np.NINF] = 0
    ir_signal = torch.Tensor(ir_signal[-5:]).to(device)
    ir_signal = ir_signal.view(size=(1, ir_signal.size(0)))
    return ir_signal

def train_classifier_network(network, train_dataset, epochs, device, learning_rate=0.01):
    total_iterations = 0
    total_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_dataset)):

            optimizer.zero_grad()
            outs = network(inputs)
            loss = criterion(outs, labels.to(device))
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


def classifier_network_testing(network, test_dataset, batches):
    corr = 0
    tot = 0
    counter = 0
    with torch.no_grad():
        for data, labels in test_dataset:
            counter += 1
            outs = network(data)
            _, predicted = torch.max(outs.data, 1)
            tot += labels.size(0)
            corr += (predicted == labels).sum().item()
    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test data: %d %%" % (counter * batches, acc))
    return acc, corr, tot
