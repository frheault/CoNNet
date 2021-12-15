#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn

from CoNNet.layers import E2EBlock


use_cuda = torch.cuda.is_available()


class BrainNetCNN_single(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size, nbr_class, nbr_tabular,
                 l1=16, l2=32, l3=128):
        super(BrainNetCNN_single, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size
        self.dropout = nn.Dropout(0.5)

        # For matrices
        self.e2econv1 = E2EBlock(self.num_channels, l2, matrix_size,
                                 bias=True)
        self.E2N = torch.nn.Conv2d(l2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, l3, (self.d, 1))
        self.dense1 = torch.nn.Linear(l3+nbr_tabular, l2)
        self.dense3 = torch.nn.Linear(l2, nbr_class)

        # for tabular data (if a lot)
        # if nbr_tabular > 0:
        #     self.tabular_dense1 = torch.nn.Linear(21, 16)

    def forward(self, x, t=None):
        out = F.leaky_relu(self.e2econv1(x),
                           negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out),
                           negative_slope=0.33)

        out = F.leaky_relu(self.N2G(out),
                           negative_slope=0.33)
        out = out.view(out.size(0), -1)

        if t is None:
            out = self.dropout(F.leaky_relu(self.dense1(out),
                                            negative_slope=0.33))
        else:
            # for tabular data (simplify, then concat)
            # tab = self.tabular_dense1(t)
            # tab = self.relu(tab)
            out = torch.cat((out, t), dim=1)
            out = self.dropout(F.leaky_relu(self.dense1(out),
                                            negative_slope=0.33))

        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out


class BrainNetCNN_double(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size, nbr_class, nbr_tabular,
                 l1=16, l2=32, l3=128):
        super(BrainNetCNN_double, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size
        self.dropout = nn.Dropout(0.5)

        # For matrices
        self.e2econv1 = E2EBlock(self.num_channels, l1, matrix_size,
                                 bias=True)
        self.e2econv2 = E2EBlock(l1, l2, matrix_size,
                                 bias=True)
        self.E2N = torch.nn.Conv2d(l2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, l3, (self.d, 1))
        self.dense1 = torch.nn.Linear(l3, l2)
        self.dense2 = torch.nn.Linear(l2+nbr_tabular, l1)
        self.dense3 = torch.nn.Linear(l1, nbr_class)

        # for tabular data (if a lot)
        # if nbr_tabular > 0:
        #     self.tabular_dense1 = torch.nn.Linear(21, 16)

    def forward(self, x, t=None):
        out = F.leaky_relu(self.e2econv1(x),
                           negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out),
                           negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out),
                           negative_slope=0.33)

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
