import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

# import torch.optim as optim
# import GPUtil
# import threading
# import time
import pickle
import csv
import argparse
import os

import cornersearch_attacks_pt
import pgd_attacks_pt
from sparsefool import sparsefool
import foolbox
from foolbox.models import PyTorchModel

import time
import sys

transform = transforms.Compose([transforms.ToTensor()])

# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                       download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
# shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=512,
#                                          shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
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


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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


def adv_batch_acc(images, labels, args):
    net.eval()
    correct = 0

    images = images.permute(0, 2, 3, 1)

    total = labels.size(0)

    if args['attack'] == 'CS':
        with torch.no_grad():
            images, labels = images.numpy(), labels.numpy()
            attack = cornersearch_attacks_pt.CSattack(net, args)
            correct = attack.perturb(images, labels)

    elif args['attack'] == 'PGD':
        with torch.no_grad():
            images, labels = images.numpy(), labels.numpy()
            attack = pgd_attacks_pt.PGDattack(net, args)
            correct = attack.perturb(images, labels)

    elif args['attack'] == 'SF':
        images, labels = images.to(device), labels.to(device)
        for i in range(images.shape[0]):
            im = images[i:i + 1]

            lb = torch.zeros(images[0:1].shape, device=device)
            ub = torch.ones(images[0:1].shape, device=device)

            x_adv, r, pred_label, fool_label, loops = sparsefool(im, net, lb, ub, args['lambda_'], args['max_iter'],
                                                                 device=device)

            l0_dist = torch.sum(r != 0).item()
            if fool_label == labels[i] or (pred_label == labels[i] and l0_dist > args['sparsity']):
                correct += 1

    elif args['attack'] == 'PA':

        fmodel = PyTorchModel(net, bounds=(0, 1), device=device, num_classes=10)
        attack = foolbox.attacks.PointwiseAttack(fmodel)

        with torch.no_grad():
            images, labels = images.numpy(), labels.numpy()
            adv = attack(images, labels)

            l0_dist = np.sum((adv - images) != 0, axis=(1, 2, 3))
            imperceptable_adv = (l0_dist <= args['sparsity'])

            images[imperceptable_adv] = adv[imperceptable_adv]

            outputs = net(torch.tensor(images).to(device))
            _, predicted = torch.max(outputs.data, 1)

            correct = (predicted == torch.tensor(labels).to(device)).sum().item()

    return correct, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    # parser.add_argument('--kappa', type=float, default=0.8)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--n_examples', type=int, default=1000)
    parser.add_argument('--n_max', type=int, default=24)
    # parser.add_argument('--load_model', type=str, default="")
    # parser.add_argument('--', type=int, default=1000)
    parser.add_argument('--attack', type=str, default="PA")
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--lambda_', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--process', type=int, default=0)

    args = parser.parse_args()

    lambda_vals = [0.4, 0.5, 0.6]
    num_maxs = [24, 30, 50]
    num_examples = [10, 20, 30]
    epoches = [9, 19, 29, 39]
    batch_length = 20

    designated_models = [
        {'lambda_val': 0.5,
         'num_max': 24,
         'num_examples': 20},

        {'lambda_val': 0.5,
         'num_max': 50,
         'num_examples': 30},

        {'lambda_val': 0.5,
         'num_max': 24,
         'num_examples': 10},

        {'lambda_val': 0.4,
         'num_max': 30,
         'num_examples': 20}
    ]

    # Hyperparameters
    n_classes = 10
    n_corners = 2
    # kappa = args.kappa
    k = args.k

    process = args.process

    batch_num = process % batch_length
    process = process // batch_length

    temp_index = process % len(epoches)
    epoch = epoches[temp_index]
    process = process // len(epoches)

    # temp_index = process % len(num_examples)
    # n_iter = num_examples[temp_index]
    # process = process // len(num_examples)
    #
    # temp_index = process % len(num_maxs)
    # n_max = num_maxs[temp_index]
    # process = process // len(num_maxs)
    #
    # temp_index = process % len(lambda_vals)
    # lambda_val = lambda_vals[temp_index]
    # process = process // len(lambda_vals)

    temp_index = process % len(designated_models)
    designated_model = designated_models[temp_index]
    process = process // len(designated_models)

    n_iter = designated_model['num_examples']
    n_max = designated_model['num_max']
    lambda_val = designated_model['lambda_val']

    print("lambda val =" + str(lambda_val))
    print("n max =" + str(n_max))
    print("n iter =" + str(n_iter))
    print("batch =" + str(batch_num))
    print("epoch =" + str(epoch))
    print("Attack =" + str(args.attack))
    print("--------------------")

    if args.attack == 'CS':
        net = Net1()
        attack_args = {'attack': 'CS',
                       'type_attack': 'L0',
                       'n_iter': 1000,
                       'n_max': args.n_max,
                       'kappa': -1,
                       'epsilon': -1,
                       'sparsity': args.k,
                       'size_incr': 1}



    elif args.attack == 'PGD':
        net = Net1()
        attack_args = {'attack': 'PGD',
                       'type_attack': 'L0',
                       'n_restarts': 5,
                       'num_steps': 100,
                       'step_size': 120000.0 / 255.0,
                       'kappa': -1,
                       'epsilon': -1,
                       'sparsity': args.k}

    elif args.attack == 'SF':
        net = Net2()
        attack_args = {'attack': 'SF',
                       'max_iter': args.max_iter,
                       'lambda_': args.lambda_,
                       'sparsity': args.k}

    elif args.attack == 'PA':
        net = Net2()
        attack_args = {'attack': 'PA',
                       'sparsity': args.k}

    net = nn.DataParallel(net)
    net = net.to(device)

    train_directory = "../train/mnist" + "/l_" + str(lambda_val) + "_N_" + str(n_max) + "_e_" + str(
        n_iter) + "/models"

    path = os.path.join(train_directory, "e_" + str(epoch) + ".pth")

    #### check if results exist
    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    par_dir = os.path.join(par_dir, "eval")

    file_name = args.attack + "/epoch_" + str(epoch) + "_batch_" + str(batch_num) + ".csv"
    if os.path.isfile(os.path.join(par_dir, file_name)):
        print("Results have already existed!")
        sys.exit()

    #### Further Execution
    net.load_state_dict(torch.load(path))

    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    par_dir = os.path.join(par_dir, "eval")
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)

    b, e = batch_num * args.batch_size, min(len(testset), (batch_num + 1) * args.batch_size)
    images = torch.stack([testset[i][0] for i in range(b, e)])
    labels = torch.tensor([testset[i][1] for i in range(b, e)])

    correct, total = adv_batch_acc(images, labels, attack_args)

    try:
        if not os.path.exists(os.path.join(par_dir, args.attack)):
            os.makedirs(os.path.join(par_dir, args.attack))
    except:
        print("Directory Make Error!")
        time.sleep(60)

    file_name = args.attack + "/epoch_" + str(epoch) + "_batch_" + str(batch_num) + ".csv"
    with open(os.path.join(par_dir, file_name), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch_num", "correct", "total", "accuracy"])
        writer.writerow([batch_num, correct, total, correct / total])
