import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import argparse

# from resnet import ResNet18
from utils_pt import load_data

import torch
import torchvision
import torchvision.transforms as transforms     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
import csv

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

log_info = {"epoch":0, "batch":0,"train_adv_acc":0, "train_clean_acc":0, "train_loss":0}

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


def normal_acc():
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			# images = images.permute(0, 2, 3, 1)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	loss = torch.nn.CrossEntropyLoss()
	cost = loss(outputs, labels).to(device)
	print("normal acc:\t", 100 * correct / total)

	return 100 * correct / total

def train(attack, net, num_epochs, init_epoch, init_batch, train_dir):

	for epoch in range(num_epochs):
		net.train()
		steps = 0
		running_loss = 0.0

		for i, data in enumerate(trainloader, 0):

			if i < init_batch:
				continue

			print("epoch:", init_epoch+epoch, "batch:", i)
			    
			x_nat, y_nat = data[0].to(device), data[1].to(device)
			x_nat = x_nat.permute(0, 2, 3, 1).to(device)
			optimizer.zero_grad()

			# adv = attack(x_nat, y_nat, n_max, n_iter, k)
			adv, _ = attack.perturb(x_nat, y_nat)
			adv = torch.tensor(adv, device=device).permute(0, 3, 1, 2)
			
			outputs = net(adv)
			loss = criterion(outputs, y_nat)

			loss.backward()
			optimizer.step()

			steps += 1
			running_loss += loss.item()


			print("training loss:", loss.item())
			net.eval()
			clean_acc = normal_acc()
			log_info["train_clean_acc"] = clean_acc
			log_info["epoch"] = epoch + init_epoch
			log_info["batch"] = i
			log_info["train_loss"] = loss.item()

			with open(train_dir + "train_log.csv", 'a') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(list(log_info.values()))

			print("\n")

			
		path = train_dir + "models/e_" + str(init_epoch + epoch) + ".pth"
		torch.save(net.state_dict(), path)

		file_ = open(train_dir + "train_info", 'wb')
		pickle.dump([init_epoch + epoch], file_)
		file_.close()

		print("model saved!")

		print("")



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define hyperparameters.')
	parser.add_argument('--dataset', type=str, default='mnist', help='cifar10, mnist')
	parser.add_argument('--attack', type=str, default='PGD', help='PGD, CS')
	# parser.add_argument('--path_results', type=str, default='none')
	# parser.add_argument('--n_examples', type=int, default=50)
	parser.add_argument('--train_directory', type=str, default="")
	parser.add_argument('--resume', type=bool, default=False)
	parser.add_argument('--epoch_num', type=int, default=40)
	# parser.add_argument('--data_dir', type=str, default= './data')

	args = parser.parse_args()

	# load model
	model = Net()
	model = nn.DataParallel(model)	
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()


	if args.train_directory != "":
		if not os.path.exists(args.train_directory):
			os.makedirs(args.train_directory)

	if not os.path.exists(os.path.join(args.train_directory, "models")):
		os.makedirs(os.path.join(args.train_directory, "models"))	


	init_epoch, init_batch = 0, 0

	if args.resume:
		file_ = open(os.path.join(args.train_directory, "train_info"), 'rb')
		temp = pickle.load(file_)
		file_.close()
		init_epoch, init_batch = temp[0], temp[1]+1

		path = os.path.join(args.train_directory, "models/e_" + str(temp[0]) + "_b_" + str(temp[1]) + ".pth")
		model.load_state_dict(torch.load(path))


	import pgd_attacks_pt

	attack_args = {'type_attack': 'L0',
	            'n_restarts': 5,
	            'num_steps': 40,
	            'step_size': 30000.0/255.0,
	            'kappa': -1,
	            'epsilon': -1,
	            'sparsity': 20}
	        
	attack = pgd_attacks_pt.PGDattack(model, attack_args)

	train(attack, model, args.epoch_num, init_epoch, init_batch, args.train_directory)
    
    
    # if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)  
  
  # elif hps.attack == 'CS':
  #   import cornersearch_attacks_pt
    
  #   args = {'type_attack': 'L0',
  #           'n_iter': 1000,
  #           'n_max': 100,
  #           'kappa': -1,
  #           'epsilon': -1,
  #           'sparsity': 10,
  #           'size_incr': 1}
    
  #   attack = cornersearch_attacks_pt.CSattack(model, args)
    
  #   adv, pixels_changed, fl_success = attack.perturb(x_test, y_test)
    
  #   if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)
    
