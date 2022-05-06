#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

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
        del self.cnn2.weight
        self.cnn2.weight = torch.swapaxes(self.cnn1.weight, 2, 3).clone()

    def forward(self, x):
        self.cnn2.weight = torch.swapaxes(self.cnn1.weight, 2, 3).clone()

        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3) + torch.cat([b]*self.d, 2)


class LinearBlocks(torch.nn.Module):
    '''LinearBlocks.'''

    def __init__(self, l1, l2, nbr_tabular=0, nbr_classes_each=0):
        super(LinearBlocks, self).__init__()
        self.dense1 = torch.nn.Linear(l1, l2)
        self.dense2 = torch.nn.Linear(l2+nbr_tabular, l2)
        self.dense3 = torch.nn.Linear(l2, l2)
        self.dense4 = torch.nn.Linear(l2, nbr_classes_each)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, t=None):
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
        out = F.leaky_relu(self.dense3(out),
                           negative_slope=0.33)
        out = F.leaky_relu(self.dense4(out),
                           negative_slope=0.33)
        return out

# https://amaarora.github.io/2020/09/13/unet.html
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, in_planes, matrix_size,
                 enc_chs=(64, 128, 256),
                 dec_chs=(256, 128, 64)):
        super().__init__()
        self.encoder = Encoder((in_planes,)+enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], in_planes, 1)
        self.matrix_size = matrix_size

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = F.interpolate(out, (self.matrix_size, self.matrix_size))

        return out
