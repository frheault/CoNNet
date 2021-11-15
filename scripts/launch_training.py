#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse
import os
from CoNNet.utils import load_data, ConnectomeDataset
from CoNNet.models import BrainNetCNN
import torch
import numpy as np

import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable

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
    trainset = ConnectomeDataset(loaded_stuff, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                              shuffle=True, num_workers=1)

    testset = ConnectomeDataset(loaded_stuff, mode='validation')
    testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                             shuffle=False, num_workers=1)

    net = BrainNetCNN(trainset.X)

    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    momentum = 0.9
    lr = 0.005
    wd = 0.0005  # Decay for L2 regularization
    #wd = 0


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr, momentum=momentum, nesterov=True,
                                weight_decay=wd)


    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(outputs, targets)
            
            # print statistics
            running_loss += loss.data.item()

            
            if batch_idx % 10 == 9:    # print every 10 mini-batches
               print('Training loss: %.6f' % ( running_loss / 10))
               running_loss = 0.0
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            
            correct += predicted.eq(targets.data).cpu().sum()

        return running_loss/batch_idx

    def test():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        running_loss = 0.0

        preds = []
        ytrue = []

        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)

                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.data.item()
                
                if use_cuda:
                    outputs, targets = outputs.cpu(), targets.cpu()

                preds.append(outputs.numpy())
                ytrue.append(targets.numpy())
            
            # print statistics
            running_loss += loss.data.item()
            if batch_idx % 5 == 4:    # print every 5 mini-batches
               print('Test loss: %.6f' % ( running_loss / 5))
               running_loss = 0.0
                
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        return np.hstack(preds).T, np.vstack(ytrue), running_loss/batch_idx

    nbepochs = 200
    allloss_train = []
    allloss_test= []

    allmae_test1 = []
    allpears_test1 = []

    allmae_test2 = []
    allpears_test2 = []
    for epoch in range(nbepochs):
        loss_train = train(epoch)
        allloss_train.append(loss_train)

        preds,y_true,loss_test = test()
        allloss_test.append(loss_test)
        
        print("Epoch %d" % epoch)
        print(preds.shape, y_true.shape)
        mae_1 = mae(preds[:,0],y_true[:,0])
        pears_1 = pearsonr(preds[:,0],y_true[:,0])
        
        allmae_test1.append(mae_1)
        allpears_test1.append(pears_1)
        
        print("Test Set : MAE for Engagement : %0.2f %%" % (100*mae_1))
        print("Test Set : pearson R for Engagement : %0.2f, p = %0.2f" % (pears_1[0],pears_1[1]))

        mae_2 = mae(preds[:,1],y_true[:,1])
        pears_2 = pearsonr(preds[:,1],y_true[:,1])
        
        allmae_test2.append(mae_2)
        allpears_test2.append(pears_2)
        
        print("Test Set : MAE for Training : %0.2f %%" % (100*mae_2))
        print("Test Set : pearson R for Training : %0.2f, p = %0.2f" % (pears_2[0],pears_2[1]))

if __name__ == "__main__":
    main()
