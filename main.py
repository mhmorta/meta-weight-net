# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
import sklearn.metrics as sm
import random
import numpy as np

from load_corrupted_data import CIFAR10, CIFAR100

from data import build_dataset, build_dataset_continual
from trainer import train_weighted

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=200, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)

#os.environ['CUD_DEVICE_ORDER'] = "1"
#ids = [1]



use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


def main():
    global args
    args = parser.parse_args()
    print()
    print(args)

    # train_loaders, train_meta_loader, test_loader = build_dataset(args)
    train_loaders, train_meta_loader, test_loader = build_dataset_continual(args)
    data_loaders = [train_loaders, train_meta_loader, test_loader]


    meta_model_loss, model_loss, accuracy_log, train_acc =train_weighted(args, data_loaders)

    #np.save('meta_model_loss_%s_%s.npy' % (args.dataset, args.label_corrupt_prob), meta_model_loss)
    #np.save('model_loss_%s_%s.npy' % (args.dataset, args.label_corrupt_prob), model_loss)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    ax1, ax2, ax3 = axes.ravel()

    ax1.plot(meta_model_loss, label='meta_model_loss')
    ax1.plot(model_loss, label='model_loss')
    ax1.set_ylabel("Losses")
    ax1.set_xlabel("Iteration")
    ax1.legend()

    acc_log = np.concatenate(accuracy_log, axis=0)
    train_acc_log = np.concatenate(train_acc, axis=0)
    #np.save('L2SPL_train_acc.npy', train_acc_log)
    #np.save('L2SPL_val_acc.npy', acc_log)
    # lr_log = np.concatenate(lr_log, axis=0)

    ax2.plot(acc_log[:, 0], acc_log[:, 1])
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Iteration')

    ax3.plot(train_acc_log[:, 0], train_acc_log[:, 1])
    ax3.set_ylabel('Accuracy')
    ax3.set_xlabel('Iteration')

    plt.show()


if __name__ == '__main__':
    main()
