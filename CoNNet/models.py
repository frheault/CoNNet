#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from CoNNet.layers import E2EBlock, LinearBlocks


use_cuda = torch.cuda.is_available()


class BrainNetCNN_double(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size, nbr_class, nbr_tabular,
                 l1=16, l2=32, l3=128):
        super(BrainNetCNN_double, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size
        self.dropout = nn.Dropout(0.5)

        # For matrices
        self.E2Econv1 = E2EBlock(self.num_channels, l1, matrix_size,
                                 bias=False)
        self.E2Econv2 = E2EBlock(l1, l2, matrix_size,
                                 bias=False)
        self.E2N = torch.nn.Conv2d(l2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, l3, (self.d, 1))
        self.dense1 = torch.nn.Linear(l3, l2)
        self.dense2 = torch.nn.Linear(l2+nbr_tabular, l1)
        self.dense3 = torch.nn.Linear(l1, nbr_class)

    def forward(self, x, t=None):
        out = F.leaky_relu(self.E2Econv1(x),
                           negative_slope=0.33)
        out = F.leaky_relu(self.E2Econv2(out),
                           negative_slope=0.33)
        out = self.dropout(F.leaky_relu(self.E2N(out),
                           negative_slope=0.33))

        out = self.dropout(F.leaky_relu(self.N2G(out),
                                        negative_slope=0.33))
        out = out.view(out.size(0), -1)
        out = self.dropout(F.leaky_relu(self.dense1(out),
                                        negative_slope=0.33))

        if t is None:
            out = self.dropout(F.leaky_relu(self.dense2(out),
                                            negative_slope=0.33))
        else:
            # for tabular data (simplify, then concat)
            # tab = self.tabular_dense1(t)
            # tab = self.relu(tab)
            out = torch.cat((out, t), dim=1)
            out = self.dropout(F.leaky_relu(self.dense2(out),
                                            negative_slope=0.33))

        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        # m = torch.nn.LogSoftmax(dim=1)
        # out = m(self.dense3(out))

        return out


class BrainNetCNN_double_extra(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size, nbr_classification,
                 nbr_classes_each, nbr_regression, nbr_tabular,
                 l1=16, l2=32, l3=128):
        super(BrainNetCNN_double_extra, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size
        self.dropout = nn.Dropout(0.5)

        # For matrices
        self.E2Econv1 = E2EBlock(self.num_channels, l1, matrix_size,
                                 bias=False)
        self.E2Econv2 = E2EBlock(l1, l2, matrix_size,
                                 bias=False)
        self.E2N = torch.nn.Conv2d(l2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, l3, (self.d, 1))

        self.module_list = []
        for i in range(nbr_classification):
            self.module_list.append(LinearBlocks(l3, l2, l1, nbr_tabular,
                                                 nbr_classes_each[i]))

        for i in range(nbr_regression):
            self.module_list.append(LinearBlocks(l3, l2, l1, nbr_tabular, 1))

        self.module_list = nn.ModuleList(self.module_list)
    
    def forward(self, x, t=None):
        out = F.leaky_relu(self.E2Econv1(x),
                           negative_slope=0.33)
        out = F.leaky_relu(self.E2Econv2(out),
                           negative_slope=0.33)
        out = self.dropout(F.leaky_relu(self.E2N(out),
                           negative_slope=0.33))

        out = self.dropout(F.leaky_relu(self.N2G(out),
                                        negative_slope=0.33))

        out_list = []
        for i, module in enumerate(self.module_list):
            out_list.append(F.leaky_relu(module(out, t),
                                         negative_slope=0.33))

        return torch.hstack(out_list)
