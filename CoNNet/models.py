#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn

from CoNNet.layers import E2EBlock, LinearBlocks


use_cuda = torch.cuda.is_available()


class BrainNetCNN_double(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size, nbr_classification,
                 nbr_classes_each, nbr_regression, nbr_tabular,
                 l1=16, l2=32, l3=128, l4=256):
        super(BrainNetCNN_double, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size
        self.dropout = nn.Dropout(0.5)

        # For matrices
        self.E2Econv1 = E2EBlock(self.num_channels, l1, matrix_size)
        self.E2Econv2 = E2EBlock(l1, l2, matrix_size)
        self.E2N = torch.nn.Conv2d(l2, l3, (1, self.d))
        self.N2G = torch.nn.Conv2d(l3, l4, (self.d, 1))

        self.module_list = []
        for i in range(nbr_classification):
            self.module_list.append(LinearBlocks(l4, 64, nbr_tabular,
                                                 nbr_classes_each[i]))

        for i in range(nbr_regression):
            self.module_list.append(LinearBlocks(l4, 64, nbr_tabular, 1))

        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x, t=None):
        out = self.dropout(self.E2Econv1(x))
        out = self.dropout(self.E2Econv2(out))
        out = self.dropout(F.leaky_relu(self.E2N(out),
                           negative_slope=0.33))
        out = self.dropout(F.leaky_relu(self.N2G(out),
                                        negative_slope=0.33))

        out_list = []
        for _, module in enumerate(self.module_list):
            out_list.append(F.leaky_relu(module(out, t),
                                         negative_slope=0.33))

        return torch.hstack(out_list)
