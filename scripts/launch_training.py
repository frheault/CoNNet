#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse
import os
from CoNNet.utils import (load_data, ConnectomeDataset,
                          add_noise, add_connections, remove_connections)
from CoNNet.models import BrainNetCNN
import torch
import numpy as np

import torch.nn.functional as F
import torch.nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from torchvision import transforms

import torch.backends.cudnn as cudnn
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr

use_cuda = torch.cuda.is_available()


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_folder',
                   help='')
    p.add_argument('in_labels',
                   help='')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    loaded_stuff = load_data(directory_path=args.in_folder,
                             labels_path=args.in_labels,
                             features_filename_exclude=['tot_commit2_weights.npy',
                                                        'sc_edge_normalized.npy',
                                                        'sc_vol_normalized.npy'])

    transform = transforms.Compose([add_connections(), remove_connections()])
    trainset_ori = ConnectomeDataset(loaded_stuff, mode='train',
                                     transform=False)
    trainset_add_rem = ConnectomeDataset(loaded_stuff, mode='train',
                                         transform=transform)
    trainset_noise = ConnectomeDataset(loaded_stuff, mode='train',
                                       transform=add_noise())
    trainset = ConcatDataset([trainset_ori,
                              trainset_add_rem,
                              trainset_noise])
    trainloader = DataLoader(trainset, batch_size=60,
                             shuffle=True, num_workers=1)
    testset_ori = ConnectomeDataset(loaded_stuff, mode='test',
                                     transform=False)
    testset_add_rem = ConnectomeDataset(loaded_stuff, mode='test',
                                         transform=transform)
    testset_noise = ConnectomeDataset(loaded_stuff, mode='test',
                                       transform=add_noise())
    testset = ConcatDataset([testset_ori,
                              testset_add_rem,
                              testset_noise])
    testloader = DataLoader(testset, batch_size=60,
                            shuffle=False, num_workers=1)

    # Number of features / matrix size
    nbr_features = loaded_stuff[3].shape[1]
    matrix_size = loaded_stuff[3].shape[2]
    net = BrainNetCNN(nbr_features, matrix_size)

    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    momentum = 0.9
    lr = 0.01
    wd = 0.0005  # Decay for L2 regularization
    #wd = 0

    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr, momentum=momentum)

    def train(epoch):
        net.train()
        # train_loss = 0
        # correct = 0
        # total = 0
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # targets = torch.unsqueeze(targets, dim=1).long()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets).long()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()

            # if batch_idx % 10 == 9:    # print every 10 mini-batches
            #    print('Training loss: %.6f' % ( running_loss / 10))
            #    running_loss = 0.0
            # _, predicted = torch.max(outputs.data, 1)

            # total += targets.size(0)
            # correct += predicted.eq(targets.data).cpu().sum()
        # print('train', running_loss/batch_idx, correct/total)
        return running_loss/batch_idx

    def test():
        net.eval()
        test_loss = 0
        # correct = 0
        # total = 0
        running_loss = 0.0

        preds = []
        ytrue = []

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets).long()

                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.data.item()

                if use_cuda:
                    outputs, targets = outputs.cpu(), targets.cpu()
                
                # print('&&&&', outputs.shape, targets.shape)
                # preds.append(torch.max(outputs.data, 1)[1].numpy())
                preds.append(outputs.numpy())
                ytrue.append(targets.numpy())

            # print statistics
            running_loss += loss.data.item()
            # if batch_idx % 10 == 9:    # print every 5 mini-batches
            #     print('Test loss: %.6f' % (running_loss / 5))
            #     running_loss = 0.0

            # _, predicted = torch.max(outputs.data, 1)

            # total += targets.size(0)
            # correct += predicted.eq(targets.data).cpu().sum()

        return np.vstack(preds), np.hstack(ytrue), running_loss/batch_idx

    nbepochs = 250
    allloss_train = []
    allloss_test = []

    for epoch in range(nbepochs):
        loss_train = train(epoch)
        allloss_train.append(loss_train)

        preds, y_true, loss_test = test()
        allloss_test.append(loss_test)

        print("Epoch %d" % epoch)
        for i in range(len(preds)):
            print(i, preds[i].argmax(), int(y_true[i]))
        print()


if __name__ == "__main__":
    main()
