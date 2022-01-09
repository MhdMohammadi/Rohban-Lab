import argparse
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms     

import torch.nn as nn
import torch.nn.functional as F

import pickle
import csv
import os

from datetime import datetime

# import GPUtil
# import threading
# import time

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_info = {"epoch":0, "batch":0, "train_adv_acc":0, "train_clean_acc":0, "train_loss":0}

# def show_gpu_usage():
#   while True:
#       time.sleep(60)
#        GPUtil.showUtilization()

def print_time(message):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(message, current_time)


def train(net, num_epochs, init_epoch, init_batch, train_dir):
    global criterion
	
    counter = 0
    print_time("rigth before forward pass")
    for epoch in range(num_epochs):
        
        for i, data in enumerate(trainloader, 0):

        	if counter >= 8 * 32 * 32:
        		break
                
        	x_nat, y_nat = data[0].to(device), data[1].to(device)
        	x_nat = x_nat.permute(0, 2, 3, 1).to(device)

            
        	outputs = net(x_nat)
        
    print_time("forward pass done")


def normal_acc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.permute(0, 2, 3, 1)

            # Remove permute
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("normal acc:\t", 100 * correct / total)

    return 100 * correct / total



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, CIFAR10')
    parser.add_argument('--net_arch', type=str, default='Conv2Net', help='Conv2Net, ResNet18, ResNet50')
    parser.add_argument('--k', type=int, default=10)
    # parser.add_argument('--n_examples', type=int, default=20)
    # parser.add_argument('--n_max', type=int, default=24)
    # parser.add_argument('--lambda_val', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam, sgd')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--train_directory', type=str, default=".")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--load_model', type=str, default="")

    #Uncomment if attack added. 
    # parser.add_argument('--attack', type=bool, default=False)
    # parser.add_argument('--attack_batch_num', type=int, default=1)

    #Currently not using values from args.
    #Change These for different datasets.
    lambda_vals = [0.5]

    #Changed for CIFAR10
    num_maxs = [50, 100]
    num_examples = [10, 50]


    args = parser.parse_args()

    # Ashkan
    # trainloader, testloader, n_classes = utils.dataset_loader(args.dataset, batch_size=64)

    trainloader, testloader, n_classes = utils.dataset_loader(args.dataset)
    n_channels= next(iter(trainloader))[0].shape[1]
    n_corners = 2 ** n_channels
    k = args.k

    # os.makedirs(args.train_directory, exist_ok=True)

    lambda_val, n_max, n_iter= None, None, None

    for l_val in lambda_vals:
        for num_max in num_maxs:
            for num_example in num_examples:

                print_time("execution started")


                lambda_val = l_val
                n_max = num_max
                n_iter = num_example

                train_directory = os.path.join(args.train_directory, "l_" + str(lambda_val) + "_N_" + str(n_max) + "_e_" + str(n_iter))

                net = utils.net_loader(args.net_arch, n_channels)
                net = nn.DataParallel(net)	
                net = net.to(device)

                # os.makedirs(train_directory, exist_ok=True)
                os.makedirs(os.path.join(train_directory, "models"), exist_ok=True)	

                # t1 = threading.Thread(target=show_gpu_usage)
                # t1.start()

                init_epoch, init_batch = 0, 0

                if args.resume:
                    if os.path.exists(os.path.join(train_directory, "train_info")):
                        file_ = open(os.path.join(train_directory, "train_info"), 'rb')
                        temp = pickle.load(file_)
                        file_.close()

                        init_epoch = temp[0]+1

                        # Uncommnet if you saved models after each batch.
                        # init_epoch, init_batch = temp[0], temp[1]+1

                        path = os.path.join(train_directory, "models/e_" + str(temp[0]) + "_b_" + str(temp[1]) + ".pth")
                        net.load_state_dict(torch.load(path))

                if args.load_model != "":
                    net.load_state_dict(torch.load(args.load_model))


                # bb = BlackBox_distributer()

                optimizer = utils.optimizer_loader(net.parameters(), args.optimizer, args.lr)
                criterion = nn.CrossEntropyLoss()

                train(net, args.epochs, init_epoch, init_batch, train_directory)


