import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, 100)
        self.lin2 = torch.nn.Linear(100, output_size)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return self.softmax(data)
