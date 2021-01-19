import torch
import torch.nn.functional as functional


class CNN(torch.nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), padding=1)
        self.conv1 = torch.nn.Conv2d(3, 16, (4, 4), stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 4, (3, 3), padding=1)
        self.conv3 = torch.nn.Conv2d(4, 1, (3, 3))
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.linear = torch.nn.Linear(1024, output_size)
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
        # x = self.conv3(x)
        # x = functional.relu(x)
        # x = self.max_pool(x)
        # print(x.size())
        x = x.view(-1, 1024)
        x = self.linear(x)
        return self.softmax(x)
