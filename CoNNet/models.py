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
    def __init__(self, nbr_channels, matrix_size):
        super(BrainNetCNN, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size

        self.e2econv1 = E2EBlock(self.num_channels, 16, matrix_size,
                                 bias=True)
        self.e2econv2 = E2EBlock(16, 32, matrix_size,
                                 bias=True)
        self.E2N = torch.nn.Conv2d(32, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 128, (self.d, 1))
        self.dense1 = torch.nn.Linear(128, 64)
        self.dense2 = torch.nn.Linear(64, 16)
        self.dense3 = torch.nn.Linear(16, 5)

    def forward(self, x):
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
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out
