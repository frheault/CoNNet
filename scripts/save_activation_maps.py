#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse
from functools import partial
import os
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt

import coloredlogs

use_cuda = torch.cuda.is_available()

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_model',
                   help='Path to a trained models.')
    p.add_argument('out_dir',
                   help='Output directory.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    net = torch.load(args.in_model)
    if use_cuda:
        net = net.cuda(0)

    data = np.rollaxis(np.load('merged.npy').astype(np.float32), axis=-1)
    data = torch.tensor(data).cuda()
    print(data.dtype)
    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    data = data.unsqueeze_(0)

    for name, module in net.named_modules():
        if 'cnn' in name:
            module.register_forward_hook(get_activation(name))
            print('a', name)
        elif name in ['E2N', 'N2G']:
            print('b', name)
            print(module.weight.shape)
    output = net(data)
    for name, module in net.named_modules():
        if 'cnn' in name:
            act = activation[name]
            print(act.shape)
            fig, axarr = plt.subplots(act.size(0))
            for idx in range(act.size(0)):
                axarr[idx].imshow(act[idx])
    plt.show()

if __name__ == "__main__":
    main()
