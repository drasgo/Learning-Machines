import torch
import torch.nn.functional as functional


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3), padding=1)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 4, (3, 3), padding=1)
        self.linear = torch.nn.Linear(196, 10)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        """

        :param x:

        """
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = functional.relu(x)
        x = self.max_pool(x)
        x = x.view(-1, 196)
        x = self.linear(x)
        return self.softmax(x)
