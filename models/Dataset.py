import torch


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, inps, labels, transform=None):
        """Initialization"""
        self.labels = labels
        self.inps = inps
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inps)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        x = self.inps[index]
        if self.transform is not None:
            x = self.transform(x)
        y = int(self.labels[index])

        return x, y


