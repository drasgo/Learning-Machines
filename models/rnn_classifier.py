import torch


class RNN(torch.nn.Module):
    # TODO
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return self.softmax(data)
