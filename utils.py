import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.Dataset import Dataset


def get_ir_signal(rob, device):
    ir_signal = np.log(np.array(rob.read_irs())) / 10
    ir_signal[ir_signal == np.NINF] = 0
    ir_signal = torch.Tensor(ir_signal[-5:]).to(device)
    ir_signal = ir_signal.view(size=(1, ir_signal.size(0)))
    return ir_signal


def prepare_dataset(train_data, train_pred, batch_size, transform=None):
    trainset = Dataset(train_data, train_pred, transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def prepare_mlp_datasets(
        dataset,
        batch_size: int,
        device,
        train_size: int = 500000,
        testing_percentage: int=0.15,
        validation_percentage: int=0.15
):
    train_data, train_pred = dataset.generate_dataset(train_size)
    validation_data, validation_pred = dataset.generate_dataset(int(train_size*validation_percentage))
    test_data, test_pred = dataset.generate_dataset(int(train_size*testing_percentage))
    train_data = torch.tensor(train_data, device=device)

    train_pred= torch.tensor(train_pred, device=device)
    trainloader = prepare_dataset(train_data, train_pred, batch_size)

    if validation_percentage != 0.0:
        validation_data = torch.tensor(validation_data, device=device)
        validation_pred = torch.tensor(validation_pred, device=device)
        validation_loader = prepare_dataset(validation_data, validation_pred, batch_size)
    else:
        validation_loader = None

    if testing_percentage != 0.0:
        test_data = torch.tensor(test_data, device=device)
        test_pred = torch.tensor(test_pred, device=device)
        test_loader = prepare_dataset(test_data, test_pred, batch_size)
    else:
        test_loader = None

    return trainloader, validation_loader, test_loader


def prep_images_datasets(
        datasets_folder,
        batch_size,
        validation_percentage,
        testing_percentage,
        transform,
        image_dataset,
        images_folder,
        data_package="",
        labels_package="",
):
    data_package_path = datasets_folder  + data_package
    labels_package_path = datasets_folder + labels_package
    if data_package == "" or labels_package == "" or \
            not os.path.exists(data_package_path) or \
            not os.path.exists(labels_package_path):
        _, _ = image_dataset.generate_dataset(images_folder)

    with open(labels_package_path, "rb") as fp:
        labels = pickle.load(fp)
        assert isinstance(labels, list)
    print("loaded labels")

    with open(data_package_path, "rb") as fp:
        data = pickle.load(fp)
        assert isinstance(data, list)
    print("loaded data")

    train_data = data[int((validation_percentage + testing_percentage)* len(data)):]
    train_labels = labels[int((validation_percentage + testing_percentage) * len(labels)):]
    print("divided train data")

    validation_data = data[:int(validation_percentage * len(data))]
    validation_labels = labels[: int(validation_percentage * len(data))]
    print("divided validation data")

    testing_data = data[int(validation_percentage*len(data)): int(validation_percentage*len(data)) + int(testing_percentage * len(data))]
    testing_labels = labels[int(validation_percentage*len(labels)): int(validation_percentage*len(labels)) + int(testing_percentage * len(labels))]
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


def save_network(network, network_name: str="network.pt"):
    torch.save(network.state_dict(), network_name)


def retrieve_network(Model, device, input_nodes: int=None, hidden_nodes: int=None, output_nodes: int=6, network_name: str="network.pt"):
    if input_nodes is not None and hidden_nodes is not None:
        model = Model(input_nodes, hidden_nodes, output_nodes).to(device)
    else:
        model = Model(output_nodes).to(device)
    if os.path.exists(network_name):
        model.load_state_dict(torch.load(network_name))
        model.eval()
    return model


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
            if batch_idx % 1000 == 999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 1000)
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


def save_list_to_pickle_file(data_list, name):
    with open("datasets/pickled_datasets/" + name + ".pkl", "wb") as fp:
        pickle.dump(data_list, fp)
