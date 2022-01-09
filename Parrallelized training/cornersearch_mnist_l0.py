import argparse
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
# import GPUtil
import threading
import time
import pickle
import csv
import os

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datatsets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_info = {"epoch": 0, "batch": 0, "train_adv_acc": 0, "train_clean_acc": 0, "train_loss": 0}


# def show_gpu_usage():
# 	while True:
# 		time.sleep(60)
# 		GPUtil.showUtilization()


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


def onepixel_perturbation_logits(orig_x):
    ''' returns logits of all possible perturbations of the images orig_x
    	for each image, first comes all zeros and then all ones'''

    with torch.no_grad():
        n_corners = 2

        dims = orig_x.shape
        pic_size = dims[1] * dims[2]
        n_perturbed = pic_size * n_corners

        logits = torch.zeros(dims[0], n_perturbed, n_classes).to(device)
        for i in range(n_corners):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    perturbed = torch.clone(orig_x)
                    perturbed[:, j, k] = int(i)
                    pic_num = pic_size * i + j * dims[1] + k
                    logits[:, pic_num, :] = net(perturbed)

    return logits


def flat2square(ind):
    ''' returns the position and the perturbation given the index of an image
      of the batch of all the possible perturbations '''

    t = ind // 784
    c = (ind % 784) % 28
    r = ((ind % 784) - c) // 28

    return r, c, t


def npixels_perturbation(orig_x, dist, pixel_num):
    ''' creates n_iter images which differ from orig_x in at most k pixels '''

    with torch.no_grad():
        ind2 = torch.rand(dist.shape) + 1e-12
        ind2 = ind2.to(device)
        ind2 = torch.log(ind2) * (1 / dist)

        batch_x = orig_x.clone()
        ind_prime = torch.topk(ind2, pixel_num, 1).indices

        p11, p12, d1 = flat2square(ind_prime)
        d1 = d1.unsqueeze(2).to(device)

        counter = torch.arange(0, orig_x.shape[0])

        for i in range(orig_x.shape[0]):
            batch_x[i, p11[i], p12[i]] = d1[i].float()

    return batch_x


def ranker(logit_2, batch_y):
    counter = torch.arange(0, logit_2.shape[0])
    t1 = torch.clone(logit_2[counter, :, batch_y])

    logit_2[counter, :, batch_y] = -1000.0 * torch.ones(logit_2.shape[1]).to(device)

    t2 = torch.max(logit_2, dim=2).values
    t3 = t1 - t2

    # We need to minimize over this.
    logit_3 = torch.unsqueeze(t1, 2).repeat(1, 1, n_classes) - logit_2
    logit_3[counter, :, batch_y] = t3

    for i in range(logit_3.shape[0]):
        for j in range(logit_3.shape[2]):
            temp = logit_3[i, :, j].reshape(2, -1)
            temp = torch.cat((torch.min(temp, dim=0).indices, torch.max(temp, dim=0).indices), -1)
            logit_3[i, torch.where(temp)[0], j] = 1000.0

    sorted = torch.argsort(logit_3, axis=1)
    ind = torch.zeros(logit_3.shape).to(device)

    base_dist = torch.zeros(ind.shape[1]).to(device).float()
    base_dist[:n_max] = torch.tensor([((2 * n_max - 2 * i + 1) / n_max ** 2) for i in range(1, n_max + 1)])

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
    def backward(ctx, grad_output):
        logit_2, dist = ctx.saved_tensors
        logit_prime = logit_2 + lambda_val * grad_output
        dist_prime = ranker(logit_prime)
        grad = -(ind - dist_prime) / (lambda_val + 1e-8)
        return grad


def attack(x_nat, y_nat, n_max_, n_iter_, k_):
    adv = torch.clone(x_nat).to(device)

    logit_2 = onepixel_perturbation_logits(x_nat)

    ##Creates the distribution
    dist = bb.apply(logit_2, y_nat)

    found_adv = torch.zeros(x_nat.shape[0]).to(device)
    for c2 in range(n_classes):
        for i in range(n_iter_):
            dist_cl = torch.clone(dist[:, :, c2])
            batch_x = npixels_perturbation(x_nat, dist_cl, int(k - i * (k // n_iter_)))

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

            adv = attack(x_nat, y_nat, n_max, n_iter, k)

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

    # print("testing epoch", epoch)
    # n_acc = normal_acc()
    # a_acc = adv_acc(test_batch_num)
    # with open(train_dir + "test_log.csv", 'a') as csvfile:
    # 	writer = csv.writer(csvfile)
    # 	writer.writerow([epoch, n_acc, a_acc])
    # print("")


# def adv_acc(batch_num, n_iter=1000):
# 	correct = 0
# 	total = 0

# 	net.eval()
# 	i = 0 
# 	with torch.no_grad():
# 		for data in testloader:
# 			images, labels = data[0].to(device), data[1].to(device)
# 			images = images.permute(0, 2, 3, 1)

# 			adv = attack(images, labels, 80, 1000, 50)

# 			outputs = net(adv)
# 			_, predicted = torch.max(outputs.data, 1)
# 			total += labels.size(0)
# 			correct += (predicted == labels).sum().item()

# 			i += 1
# 			if i >= batch_num:
# 				break


# 	print("adv acc:\t", 100 * correct / total)

# 	return 100 * correct / total


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

    loss = torch.nn.CrossEntropyLoss()
    cost = loss(outputs, labels).to(device)
    print("normal acc:\t", 100 * correct / total)

    return 100 * correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    # parser.add_argument('--kappa', type=float, default=0.8)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--lambda_val', type=float, default=0.5)
    parser.add_argument('--n_examples', type=int, default=20)
    parser.add_argument('--n_max', type=int, default=24)
    parser.add_argument('--resume', type=bool, default=False)
    # parser.add_argument('--attack', type=bool, default=False)
    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--train_directory', type=str, default="")
    parser.add_argument('--load_model', type=str, default="")
    # parser.add_argument('--attack_batch_num', type=int, default=1)

    lambda_vals = [0.4, 0.5, 0.6]
    num_maxs = [24, 30, 50]
    num_examples = [10, 20, 30]

    args = parser.parse_args()

    # Hyperparameters
    n_classes = 10
    n_corners = 2
    # kappa = args.kappa
    k = args.k

    if args.train_directory != "":
        if not os.path.exists(args.train_directory):
            os.makedirs(args.train_directory)

    for l_val in lambda_vals:
        for num_max in num_maxs:
            for num_example in num_examples:

                lambda_val = l_val
                n_max = num_max
                n_iter = num_example
                train_directory = args.train_directory + "l_" + str(lambda_val) + "_N_" + str(n_max) + "_e_" + str(
                    n_iter) + "/"

                net = Net()
                net = nn.DataParallel(net)
                net = net.to(device)

                if not os.path.exists(train_directory):
                    os.makedirs(train_directory)

                if not os.path.exists(train_directory + "models"):
                    os.makedirs(train_directory + "models")

                # t1 = threading.Thread(target=show_gpu_usage)
                # t1.start()

                init_epoch, init_batch = 0, 0

                if args.resume:
                    file_ = open(train_directory + "train_info", 'rb')
                    temp = pickle.load(file_)
                    file_.close()
                    init_epoch, init_batch = temp[0], temp[1] + 1

                    path = train_directory + "models/e_" + str(temp[0]) + "_b_" + str(temp[1]) + ".pth"
                    net.load_state_dict(torch.load(path))

                if args.load_model != "":
                    net.load_state_dict(torch.load(args.load_model))

                # if args.attack:
                # 	normal_acc()
                # 	adv_acc(args.attack_batch_num)

                # 	exit()

                bb = BlackBox_distributer()

                optimizer = optim.Adam(net.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()

                train(net, args.epoch_num, init_epoch, init_batch, train_directory)
