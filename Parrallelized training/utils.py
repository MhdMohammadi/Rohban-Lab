import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms     

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as torch_models

import models
import resnet
import resnet2

#TODO: Add cifar & HMNIST
def dataset_loader(dataset, batch_size=512, num_workers=8):

    if dataset == 'MNIST':

        # if data == 'mnist':
        #     transforms_channel = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        #
        # elif data == 'svhn' or data == 'cifar10':
        #     transforms_channel = transforms.Lambda(lambda x: x)

        transform = transforms.Compose([transforms.ToTensor()])

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        n_classes = 10



    elif dataset == 'CIFAR10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        n_classes = 10


    return trainloader, testloader, n_classes



def net_loader(net_arch, channels=1):
    if net_arch == 'Conv2Net':
        return models.Conv2Net(channels)

    elif net_arch == 'ResNet18':
        # return resnet2.ResNet18()
        # return resnet.ResNet18()
        return torch_models.resnet18(num_classes=10)

    elif net_arch == 'ResNet50':
        return resnet.ResNet50()

    else:
        print("No such model exists.")



def optimizer_loader(params, name, lr):
    opt=None

    if name == 'adam':
        opt = optim.Adam(params, lr=lr)

    elif name == 'sgd':
        opt = optim.SGD(params, lr=lr, momentum=0.9)

    else:
        print("No such optimizer exists.")

    return opt