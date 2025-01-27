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

# import GPUtil
# import threading
# import time

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_info = {"epoch": 0, "batch": 0, "train_adv_acc": 0, "train_clean_acc": 0, "train_loss": 0}


# def show_gpu_usage():
#   while True:
#       time.sleep(60)
#        GPUtil.showUtilization()


def onepixel_perturbation_logits(orig_x):
    ''' returns logits of all possible perturbations of the images orig_x
        for each image, first comes all zeros and then all ones, so on'''
    '''output shape: (batch_size, perturbation #, n_classes)'''
    global n_classes, n_corners

    with torch.no_grad():

        dims = orig_x.shape
        pic_size = dims[1] * dims[2]
        n_perturbed = pic_size * n_corners

        logits = torch.zeros(dims[0], n_perturbed, n_classes).to(device)

        for i in range(n_corners):
            if orig_x.shape[-1] == 1:
                pixel_val = int(i)
            elif orig_x.shape[-1] == 3:
                pixel_val = torch.tensor([i // 4, (i % 4) // 2, i % 2]).to(device)

            for j in range(dims[1]):
                for q in range(dims[2]):
                    perturbed = torch.clone(orig_x)
                    perturbed[:, j, q] = pixel_val
                    pic_num = pic_size * i + j * dims[1] + q
                    logits[:, pic_num, :] = net(perturbed)

    return logits


# TODO: Check whether it works ok
def flat2square(ind, im_shape):
    ''' returns the position and the perturbation given the index of an image
      of the batch of all the possible perturbations '''
    im_size = im_shape[0] * im_shape[1]

    t = ind // im_size
    c = (ind % im_size) % im_shape[1]
    r = ((ind % im_size) - c) // im_shape[0]

    # row, columnt, perturbation #
    return r, c, t


def npixels_perturbation(orig_x, dist, pert_size):
    '''dist shape: (batch_size, perturbation #)
    output shape = orig_x shape
    creates a batch of images (given a batch), each differs pert_size pixels from the original.'''

    with torch.no_grad():
        ind2 = torch.rand(dist.shape) + 1e-12
        ind2 = ind2.to(device)
        ind2 = torch.log(ind2) * (1 / dist)

        batch_x = orig_x.clone()
        # ind_prime shape: (batch_size, pert_size)
        ind_prime = torch.topk(ind2, pert_size, 1).indices

        p11, p12, d1 = flat2square(ind_prime, orig_x.shape[1:])
        d1 = d1.unsqueeze(2).to(device)

        counter = torch.arange(0, orig_x.shape[0])

        # TODO: are p11, p12 arrays? does the line below work?
        for i in range(orig_x.shape[0]):
            batch_x[i, p11[i], p12[i]] = d1[i].float()

    return batch_x


def ranker(logit_2, batch_y):
    '''output shape = logit_2 shape = (batch_size, perturbation #, n_classes) '''
    global n_corners, n_max

    counter = torch.arange(0, logit_2.shape[0])

    # t1 shape: (batch_size, perturbation #)
    t1 = torch.clone(logit_2[counter, :, batch_y])

    logit_2[counter, :, batch_y] = -1000.0 * torch.ones(logit_2.shape[1]).to(device)

    t2 = torch.max(logit_2, dim=2).values
    t3 = t1 - t2

    # In logit_3, Lower score indicates better fooling.
    logit_3 = torch.unsqueeze(t1, 2).repeat(1, 1, n_classes) - logit_2
    logit_3[counter, :, batch_y] = t3

    # Make sure a single pixel (with different perturbations) is not selected more than once.
    for i in range(logit_3.shape[0]):
        for j in range(logit_3.shape[2]):
            temp = logit_3[i, :, j].reshape(n_corners, -1)
            best_pert = torch.ones(temp.shape).to(device)
            best_idx = torch.argmin(temp, dim=0)

            counter = torch.arange(0, temp.shape[1])
            best_pert[best_idx, counter] = 0
            logit_3[i, torch.where(best_pert)[0], j] = 1000.0

    # Not so efficient
    ''' sorted = torch.argsort(logit_3, axis=1)

    ind  = torch.zeros(logit_3.shape).to(device)


    base_dist = torch.zeros(ind.shape[1]).to(device).float()
    base_dist[:n_max] = torch.tensor([((2*n_max - 2*i + 1)/n_max**2) for i in range(1, n_max+1)])'''

    # Please check this
    sorted = torch.topk(input=logit_3, k=n_max, dim=1, largest=False, sorted=True).indices.to(device)

    ind = torch.zeros(logit_3.shape).to(device)
    base_dist = torch.zeros(n_max).to(device).float()
    base_dist = torch.tensor([((2 * n_max - 2 * i + 1) / n_max ** 2) for i in range(1, n_max + 1)]).to(device)

    for i in range(ind.shape[0]):
        for j in range(ind.shape[2]):
            ind[i, sorted[i, :, j], j] = base_dist

    return ind


class BlackBox_distributer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logit_2, batch_y):
        dist = ranker(logit_2, batch_y)
        ctx.save_for_backward(logit_2, dist)
        return dist

    @staticmethod
    def backward(catx, grad_output):
        global lambda_val

        logit_2, dist = ctx.saved_tensors
        logit_prime = logit_2 + lambda_val * grad_output
        dist_prime = ranker(logit_prime)
        grad = -(dist - dist_prime) / (lambda_val + 1e-8)
        return grad


def attack(x_nat, y_nat):
    global k

    adv = torch.clone(x_nat).to(device)
    logit_2 = onepixel_perturbation_logits(x_nat)

    # Creates the distribution
    # dist shape = (batch_size, perturbation #, n_classes)
    dist = bb.apply(logit_2, y_nat)

    found_adv = torch.zeros(x_nat.shape[0]).to(device)
    for cl in range(n_classes):
        for i in range(n_iter):
            dist_cl = torch.clone(dist[:, :, cl])

            # Changin perturbations size to avoid overfitting. Preference on lower perturbation size.
            batch_x = npixels_perturbation(x_nat, dist_cl, int(k - i * (k // n_iter)))

            preds = torch.argmax(net(batch_x), dim=1)
            adv_indices = torch.nonzero(~torch.eq(preds, y_nat), as_tuple=False).flatten()

            adv[adv_indices] = batch_x[adv_indices]
            found_adv[adv_indices] = 1

    num_of_found = torch.sum(found_adv).item()
    batch_size = x_nat.shape[0]
    print("found: ", int(num_of_found), "/", batch_size)
    adv_acc = (batch_size - num_of_found) / batch_size * 100
    print("adv_acc:", adv_acc)

    log_info["train_adv_acc"] = adv_acc

    return adv


def train(net, num_epochs, init_epoch, init_batch, train_dir):
    global criterion

    for epoch in range(num_epochs):
        net.train()
        steps = 0
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            if i < init_batch:
                continue

            print("epoch:", init_epoch + epoch, "batch:", i)

            x_nat, y_nat = data[0].to(device), data[1].to(device)
            x_nat = x_nat.permute(0, 2, 3, 1).to(device)
            optimizer.zero_grad()

            adv = attack(x_nat, y_nat)

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

            with open(os.path.join(train_dir, "train_log.csv"), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list(log_info.values()))

            # Uncomment for saving after each batch. Remember to save models after each batch.
            # with open(os.path.join(train_dir, "train_info"), 'wb') as file_:
            # 	pickle.dump([init_epoch + epoch, i], file_)
            # 	file_.close()

            print("\n")

        path = os.path.join(train_dir, "models/e_" + str(init_epoch + epoch) + ".pth")
        torch.save(net.state_dict(), path)

        with open(os.path.join(train_dir, "train_info"), 'wb') as file_:
            pickle.dump([init_epoch + epoch], file_)
            file_.close()

        print("model saved!\n")


def normal_acc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.permute(0, 2, 3, 1)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("normal acc:\t", 100 * correct / total)

    return 100 * correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='MNIST, CIFAR10')
    parser.add_argument('--net_arch', type=str, default='ResNet18', help='Conv2Net, ResNet18, ResNet50')
    parser.add_argument('--k', type=int, default=10)
    # parser.add_argument('--n_examples', type=int, default=20)
    # parser.add_argument('--n_max', type=int, default=24)
    # parser.add_argument('--lambda_val', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='sgd', help='adam, sgd')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--train_directory', type=str, default="train/cifar10")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--load_model', type=str, default="")
    parser.add_argument('--process', type=int, default=0)

    #Uncomment if attack added. 
    # parser.add_argument('--attack', type=bool, default=False)
    # parser.add_argument('--attack_batch_num', type=int, default=1)

    args = parser.parse_args()

    #Currently not using values from args.
    #Change These for different datasets.
    lambda_vals = [0.5]
    ks = [15]

    #Changed for CIFAR10
    num_maxs = [50, 100]
    num_examples = [10, 50]


    process = args.process

    temp_index = process % len(lambda_vals)
    lambda_val = lambda_vals[temp_index]
    process = process // len(lambda_vals)
    
    temp_index = process % len(num_maxs)
    n_max = num_maxs[temp_index]
    process = process // len(num_maxs)
    
    temp_index = process % len(num_examples)
    n_iter = num_examples[temp_index]
    process = process // len(num_examples)

    temp_index = process % len(ks)
    k = ks[temp_index]
    process = process // len(ks)
    
    print("dataset =" + args.dataset)
    print("k =" + str(k))
    print("lambda val =" + str(lambda_val))
    print("n max =" + str(n_max))
    print("n iter =" + str(n_iter))
    print("--------------------")


    trainloader, testloader, n_classes = utils.dataset_loader(args.dataset)
    n_channels= next(iter(trainloader))[0].shape[1]
    n_corners = 2 ** n_channels
    k = args.k

    # os.makedirs(args.train_directory, exist_ok=True)

    # for l_val in lambda_vals:
    #     for num_max in num_maxs:
    #         for num_example in num_examples:

    train_directory = os.path.join(args.train_directory, "k" + str(k))
    train_directory = os.path.join(train_directory, "l_" + str(lambda_val) + "_N_" + str(n_max) + "_e_" + str(n_iter))

    net = utils.net_loader(args.net_arch, n_channels)
    net = nn.DataParallel(net)	
    net = net.to(device)

    # os.makedirs(train_directory, exist_ok=True)
    os.makedirs(os.path.join(train_directory, "models"), exist_ok=True)	

    init_epoch, init_batch = 0, 0

    if args.resume:
        if os.path.exists(os.path.join(train_directory, "train_info")):
            file_ = open(os.path.join(train_directory, "train_info"), 'rb')
            temp = pickle.load(file_)
            file_.close()

            init_epoch = temp[0] + 1

            # Uncommnet if you saved models after each batch.
            # init_epoch, init_batch = temp[0], temp[1]+1

            path = os.path.join(train_directory, "models/e_" + str(temp[0]) + "_b_" + str(temp[1]) + ".pth")
            net.load_state_dict(torch.load(path))

    if args.load_model != "":
        net.load_state_dict(torch.load(args.load_model))


    bb = BlackBox_distributer()

    optimizer = utils.optimizer_loader(net.parameters(), args.optimizer, args.lr)
    criterion = nn.CrossEntropyLoss()

    train(net, args.epochs, init_epoch, init_batch, train_directory)


