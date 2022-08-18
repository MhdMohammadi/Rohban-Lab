import torch.nn as nn
import torch.nn.functional as F


class Conv2Net(nn.Module):
    def __init__(self, channels, dataset):
        super(Conv2Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, (5, 5), stride=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=1)

        # Mohammad's Modification: 1024 for mnist, 1600 for cifar
        self.dataset = dataset
        if self.dataset == 'MNIST':
            self.fc1 = nn.Linear(1024, 1024)
        else:
            self.fc1 = nn.Linear(1600, 1024)

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        if self.dataset == 'MNIST':
            x = x.reshape(-1, 1024)
        else:
            x = x.reshape(-1, 1600)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x