import pandas as pd
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

import time
import sys

parser = argparse.ArgumentParser(description='Define hyperparameters.')
# parser.add_argument('--kappa', type=float, default=0.8)
parser.add_argument('--k', type=int, default=15)
parser.add_argument('--n_examples', type=int, default=1000)
parser.add_argument('--n_max', type=int, default=24)
# parser.add_argument('--load_model', type=str, default="")
# parser.add_argument('--', type=int, default=1000)
parser.add_argument('--attack', type=str, default="SF")
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--process', type=int, default=0)

args = parser.parse_args()

lambda_vals = [0.4, 0.5, 0.6]
num_maxs = [24, 30, 50]
num_examples = [10, 20, 30]
epoches = [9, 19, 29, 39]
batch_length = 20

# Hyperparameters
n_classes = 10
n_corners = 2
# kappa = args.kappa
k = args.k

process = args.process

temp_index = process % len(num_examples)
n_iter = num_examples[temp_index]
process = process // len(num_examples)

temp_index = process % len(num_maxs)
n_max = num_maxs[temp_index]
process = process // len(num_maxs)

temp_index = process % len(lambda_vals)
lambda_val = lambda_vals[temp_index]
process = process // len(lambda_vals)

print("lambda val =" + str(lambda_val))
print("n max =" + str(n_max))
print("n iter =" + str(n_iter))
print("--------------------")

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

# totals = list()
# corrects = list()
# for epoch in epoches:
#     total = 0
#     correct = 0
#     for batch_num in range(batch_length):
#         train_directory = "train/mnist" + "/l_" + str(lambda_val) + "_N_" + str(n_max) + "_e_" + str(
#             n_iter) + "/models"
#
#         path = os.path.join(train_directory, "e_" + str(epoch) + ".pth")
#
#         #### check if results exist
#         par_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
#         par_dir = os.path.join(par_dir, "eval")
#
#         file_name = args.attack + "/epoch_" + str(epoch) + "_batch_" + str(batch_num) + ".csv"
#         df = pd.read_csv(os.path.join(par_dir, file_name))
#         total += df['total']
#         correct += df['correct']
#
#     accuracy = correct / total
#     save_directory = "result/mnist" + "/l_" + str(lambda_val) + "_N_" + str(n_max) + "_e_" + str(
#             n_iter) + "/" + args.attack
#
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)
#
#     data = {"correct": [correct], "total": [total], "accuracy": [accuracy]}
#     result_df = pd.DataFrame.from_dict(data)
#
#     result_df.to_csv(save_directory + "/epoch_" + str(epoch) + ".csv", encoding='utf-8', index=False)

saved_lambda_vals = list()
saved_num_maxs = list()
saved_num_examples = list()
saved_epochs = list()
totals = list()
corrects = list()
accuracies = list()

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

for model in designated_models:
    lambda_val = model['lambda_val']
    n_max = model['num_max']
    n_iter = model['num_examples']
# for lambda_val in lambda_vals:
#     for n_max in num_maxs:
#         for n_iter in num_examples:
    for epoch in epoches:
        total = 0
        correct = 0
        for batch_num in range(batch_length):
            train_directory = "train/mnist" + "/l_" + str(lambda_val) + "_N_" + str(n_max) + "_e_" + str(
                n_iter) + "/models"

            path = os.path.join(train_directory, "e_" + str(epoch) + ".pth")

            #### check if results exist
            par_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            par_dir = os.path.join(par_dir, "eval")

            file_name = args.attack + "/epoch_" + str(epoch) + "_batch_" + str(batch_num) + ".csv"
            df = pd.read_csv(os.path.join(par_dir, file_name))
            total += df['total']
            correct += df['correct']

        accuracy = correct / total
        saved_lambda_vals.append(lambda_val)
        saved_num_maxs.append(n_max)
        saved_num_examples.append(n_iter)
        totals.append(int(total))
        corrects.append(int(correct))
        accuracies.append(float(accuracy))
        saved_epochs.append(epoch)


save_directory = "result/mnist/" + args.attack

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

data = {"lambda val": saved_lambda_vals, "num max": saved_num_maxs, "num example": saved_num_examples,
        "epoch": saved_epochs, "correct": corrects, "total": totals, "accuracy": accuracies}
for key in data.keys():
    print(key + str(len(data[key])))

result_df = pd.DataFrame.from_dict(data)

result_df.to_csv(save_directory + "/overall.csv", encoding='utf-8', index=False)

