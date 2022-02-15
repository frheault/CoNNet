#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn

use_cuda = torch.cuda.is_available()


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, matrix_size, bias=True):
        super(E2EBlock, self).__init__()
        self.d = matrix_size
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d),
                                    bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1),
                                    bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3) + torch.cat([b]*self.d, 2)


class LinearBlocks(torch.nn.Module):
    '''LinearBlocks.'''
    def __init__(self, l3, l2, l1, nbr_tabular=0, nbr_classes_each=0):
        super(LinearBlocks, self).__init__()
        self.dense1 = torch.nn.Linear(l3, 4096)
        self.dense2 = torch.nn.Linear(4096+nbr_tabular, l1)
        self.dense3 = torch.nn.Linear(l1, nbr_classes_each)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, t):
        out = x.view(x.size(0), -1)
        out = self.dropout(F.leaky_relu(self.dense1(out),
                                        negative_slope=0.33))

        if t is None:
            out = self.dropout(F.leaky_relu(self.dense2(out),
                                            negative_slope=0.33))
        else:
            out = torch.cat((out, t), dim=1)
            out = self.dropout(F.leaky_relu(self.dense2(out),
                                            negative_slope=0.33))

        out = self.dense3(out)
        return out


# https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440
# https://discuss.pytorch.org/t/how-to-implement-multiple-loss/45749
# https://discuss.pytorch.org/t/variable-number-of-output-branches/18244/2
# https://discuss.pytorch.org/t/help-understanding-model-output/27152
# https://discuss.pytorch.org/t/how-to-train-the-network-with-multiple-branches/2152/7
# https://discuss.pytorch.org/t/how-to-train-a-network-with-different-branches-not-parallel/49784