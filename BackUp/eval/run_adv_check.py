import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms     

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import GPUtil
import threading
import time
import pickle
import csv
import argparse

import cornersearch_attacks_pt
import pgd_attacks_pt


transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), stride=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=1)
        self.fc1 = nn.Linear(1024, 1024) 
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def adv_acc(batch_num, args):
	correct = 0
	total = 0

	net.eval()
	i = 0 
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0], data[1]
			images = images.permute(0, 2, 3, 1)

			total += labels.size(0)

			images, labels = images.numpy(), labels.numpy()

			if args['attack'] == 'CS':
				attack = cornersearch_attacks_pt.CSattack(net, args)
			
			elif args['attack'] == 'PGD':
				attack = pgd_attacks_pt.PGDattack(net, args)

			correct += attack.perturb(images, labels)
			
			
			i += 1
			if i >= batch_num:
				break

			

	print("adv acc:\t", 100 * correct / total)

	return 100 * correct / total


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define hyperparameters.')
	parser.add_argument('--kappa', type=float, default=0.8)
	parser.add_argument('--k', type=int, default=50)
	parser.add_argument('--n_examples', type=int, default=20)
	parser.add_argument('--n_max', type=int, default=80)
	parser.add_argument('--load_model', type=str, default="")
	parser.add_argument('--attack_batch_num', type=int, default=1000)
	parser.add_argument('--attack', type=str, default="CS")


	args = parser.parse_args()
	
	net = Net()
	net = nn.DataParallel(net)	
	net = net.to(device)

	if args.load_model != "":
		net.load_state_dict(torch.load(args.load_model))

	# attack_args = {'type_attack': 'L0+sigma',
	#         'n_iter': 1000,
	#         'n_max': args.n_max,
	#         'kappa': args.kappa,
	#         'epsilon': -1,
	#         'sparsity': args.k,
    #     	'size_incr': 5}

	if args.attack == 'CS':

		attack_args = {'attack': 'CS',
			'type_attack': 'L0',
			'n_iter': 1000,
			'n_max': args.n_max,
			'kappa': -1,
			'epsilon': -1,
			'sparsity': args.k,
			'size_incr': 1}
	    


	elif args.attack == 'PGD':

		attack_args = {'attack': 'PGD',
				'type_attack': 'L0',
                'n_restarts': 5,
                'num_steps': 100,
                'step_size': 120000.0/255.0,
                'kappa': -1,
                'epsilon': -1,
                'sparsity': args.k}

	adv_acc(args.attack_batch_num, attack_args)			