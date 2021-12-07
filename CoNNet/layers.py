#!/usr/bin/env python

import torch


use_cuda = torch.cuda.is_available()


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, matrix_size, bias=False):
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
