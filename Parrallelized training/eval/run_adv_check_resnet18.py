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
# from sparsefool import sparsefool
import foolbox
from foolbox.models import PyTorchModel

import resnet2

transform = transforms.Compose([transforms.ToTensor()])

# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                       download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
# shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root='../data', train=False,
                                     download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=512,
#                                          shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument('--attack', type=str, default="CS")
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--lambda_', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()

    # if args.load_model != "":
    # 	net.load_state_dict(torch.load(args.load_model))

    # attack_args = {'type_attack': 'L0+sigma',
    #         'n_iter': 1000,
    #         'n_max': args.n_max,
    #         'kappa': args.kappa,
    #         'epsilon': -1,
    #         'sparsity': args.k,
    #     	'size_incr': 5}

    net = resnet2.ResNet18()
    net = net.to(device)

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
                       'step_size': 120000.0 / 255.0,
                       'kappa': -1,
                       'epsilon': -1,
                       'sparsity': args.k}

    elif args.attack == 'SF':

        attack_args = {'attack': 'SF',
                       'max_iter': args.max_iter,
                       'lambda_': args.lambda_,
                       'sparsity': args.k}

    elif args.attack == 'PA':

        attack_args = {'attack': 'PA',
                       'sparsity': args.k}


    designated_model = [
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

    model_paths = ["../pre_trained_models/trained_pgd.pth"]
    batches = [0, 1, 2]

    for path in model_paths:
        for batch_num in batches:
            net.load_state_dict(torch.load(path))
            net = nn.DataParallel(net)

            par_dir = os.path.dirname(os.path.abspath(path))
            par_dir = os.path.join(par_dir, "eval")
            if not os.path.exists(par_dir):
                os.makedirs(par_dir)

            b, e = batch_num * args.batch_size, min(len(testset), (batch_num + 1) * args.batch_size)
            images = torch.stack([testset[i][0] for i in range(b, e)])
            labels = torch.tensor([testset[i][1] for i in range(b, e)])

            correct, total = adv_batch_acc(images, labels, attack_args)

            file_name = args.attack + "_batch_" + str(batch_num) + ".csv"
            with open(os.path.join(par_dir, file_name), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["batch_num", "correct", "total"])
                writer.writerow([batch_num, correct, total])
