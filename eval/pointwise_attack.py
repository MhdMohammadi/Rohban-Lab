import argparse
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms     

import torch.nn as nn
import torch.nn.functional as F

# import GPUtil
# import threading
# import time
import pickle
import csv
import os

import foolbox
from foolbox.models import PyTorchModel
# from foolbox.utils import accuracy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), stride=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=1)
        self.fc1 = nn.Linear(1024, 1024) 
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def adv_acc(fmodel, attack, k):
	correct = 0
	total = 0

	with torch.no_grad():
		for data in testloader:
			images, labels = data[0], data[1]
			images = images.permute(0, 2, 3, 1)

			total += labels.size(0)

			images, labels = images.numpy(), labels.numpy()

			adv = attack(images, labels)

			l0_dist = np.sum((adv - images) != 0, axis=(1, 2, 3))
			imperceptable_adv = (l0_dist <= k)

			images[imperceptable_adv] = adv[imperceptable_adv]
			
			outputs = net(torch.tensor(images).to(device))
			_, predicted = torch.max(outputs.data, 1)

			correct += (predicted == torch.tensor(labels).to(device)).sum().item()			


	print("adv acc:\t", 100 * correct / total)
	return 100 * correct / total


transform = transforms.Compose([transforms.ToTensor()])

# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                       download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                           shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Define hyperparameters.')
	parser.add_argument('--k', type=int, default=15)
	parser.add_argument('--load_model', type=str, default="")

	args = parser.parse_args()

	net = Net()
	net = nn.DataParallel(net)	
	net = net.to(device)

	if args.load_model != "":
		net.load_state_dict(torch.load(args.load_model))

	net.eval()
	fmodel = PyTorchModel(net, bounds=(0, 1), device=device, num_classes=10)
	attack = foolbox.attacks.PointwiseAttack(fmodel)

	adv_acc(fmodel, attack, args.k)






	