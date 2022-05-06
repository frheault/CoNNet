#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn
from CoNNet.layers import E2EBlock, LinearBlocks, UNet

use_cuda = torch.cuda.is_available()


class BrainNetCNN_Generator(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size):
        super(BrainNetCNN_Generator, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size
        # self.dropout = nn.Dropout(0.5)

        self.unet = UNet(nbr_channels, matrix_size)


    def forward(self, x):
        return self.unet(x)
        # out = self.dropout(self.unet(x))
        # return F.leaky_relu(out, negative_slope=0.33)


class BrainNetCNN_Discriminator(torch.nn.Module):
    def __init__(self, nbr_channels, matrix_size, num_class=2,
                 l1=16, l2=32, l3=128, l4=256):
        super(BrainNetCNN_Discriminator, self).__init__()
        self.num_channels = nbr_channels
        self.d = matrix_size
        self.dropout = nn.Dropout(0.5)

        # For matrices
        self.E2Econv1 = E2EBlock(self.num_channels, l1, matrix_size)
        self.E2Econv2 = E2EBlock(l1, l2, matrix_size)
        self.E2N = torch.nn.Conv2d(l2, l3, (1, self.d))
        self.N2G = torch.nn.Conv2d(l3, l4, (self.d, 1))

        self.Linear = LinearBlocks(l4, 64, 0, num_class)

    def forward(self, x):
        out = self.dropout(self.E2Econv1(x))
        out = self.dropout(self.E2Econv2(out))
        out = self.dropout(F.leaky_relu(self.E2N(out),
                           negative_slope=0.33))
        out = self.dropout(F.leaky_relu(self.N2G(out),
                                        negative_slope=0.33))

        out = F.leaky_relu(self.Linear(out),
                           negative_slope=0.33)

        return out
