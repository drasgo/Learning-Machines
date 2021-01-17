import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.Dataset import Dataset


def save_network(network, network_name: str="network.pt"):
    torch.save(network.state_dict(), network_name)


def retrieve_network(input_nodes:int , output_nodes: int, Model, device, network_name: str="network.pt"):
    model = Model(input_nodes, output_nodes).to(device)
    if os.path.exists("network.pt"):
        model.load_state_dict(torch.load(network_name))
        model.eval()
    return model


def prepare_datasets(dataset, batch_size: int, device, train_size: int = 500000, test_size: int=50000):
    train_data, train_pred = dataset.generate_dataset(train_size)
    test_data, test_pred = dataset.generate_dataset(test_size)

    train_data = torch.tensor(train_data, device=device)
    train_pred = torch.tensor(train_pred, device=device)
    trainset = Dataset(train_data, train_pred)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    test_data = torch.tensor(test_data, device=device)
    test_pred = torch.tensor(test_pred, device=device)
    testset = Dataset(test_data, test_pred)
    testloader = DataLoader(testset,batch_size=batch_size, shuffle=True)
    return trainloader, testloader


def get_ir_signal(rob, device):
    ir_signal = np.log(np.array(rob.read_irs())) / 10
    ir_signal[ir_signal == np.NINF] = 0
    ir_signal = torch.Tensor(ir_signal[-5:]).to(device)
    ir_signal = ir_signal.view(size=(1, ir_signal.size(0)))
    return ir_signal

