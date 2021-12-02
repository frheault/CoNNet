#!/usr/bin/env python

import torch
import numpy as np

import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
from CoNNet.layers import E2EBlock
use_cuda = torch.cuda.is_available()


class BrainNetCNN(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size, nbr_class, nbr_tabular,
                l1=16, l2=32, l3=128):
        super(BrainNetCNN, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size

        # For matrices
        self.e2econv1 = E2EBlock(self.num_channels, l1, matrix_size,
                                 bias=True)
        self.e2econv2 = E2EBlock(l1, l2, matrix_size,
                                 bias=True)
        self.E2N = torch.nn.Conv2d(l2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, l3, (self.d, 1))
        self.dense1 = torch.nn.Linear(l3, l2)
        self.dense2 = torch.nn.Linear(l2, l1)
        self.dense3 = torch.nn.Linear(l1+nbr_tabular, nbr_class)

        # for tabular data (if a lot)
        # if nbr_tabular > 0:
        #     self.tabular_dense1 = torch.nn.Linear(21, 16)

    def forward(self, x, t=None):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)

        if t is None:
            out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
            return out
        else:
            # for tabular data (simplify, then concat)
            # tab = self.tabular_dense1(t)
            # tab = self.relu(tab)
            out = torch.cat((out, t), dim=1) 
            out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

            return out
