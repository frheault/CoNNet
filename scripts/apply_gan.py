#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse
import os

import coloredlogs
from ray import tune, shutdown

from CoNNet.train_gan import (apply_gan)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_folder',
                   help='Path to subjects connectivity matrices.')
    p.add_argument('in_labels',
                   help='XLSX file with subjects informations/labels')
    p.add_argument('in_models', nargs='+',
                   help='XLSX file with subjects informations/labels')
    p.add_argument('--log_level', default='INFO',
                   help='Print extra logging information [%(default)s].')

    e = p.add_argument_group('Experiment options')
    e.add_argument('--out_folder', default='./apply/',
                   help='output folder for network application [%(default)s].')

    t2 = p.add_mutually_exclusive_group()
    t2.add_argument('--include', nargs='+',
                    help='Connectivity matrices filename to include for the '
                         'models. Exclude all but.')
    t2.add_argument('--exclude', nargs='+',
                    help='Connectivity matrices filename to exclude for the '
                         'models. Include all but.')

    m = p.add_argument_group('Model options')
    m.add_argument('--layer_1_size', nargs='+', default=[32], type=int,
                   help='List of layer (1) size to try [%(default)s].')
    m.add_argument('--layer_2_size', nargs='+', default=[64], type=int,
                   help='List of layer (2) size to try [%(default)s].')
    m.add_argument('--layer_3_size', nargs='+', default=[128], type=int,
                   help='List of layer (3) size to try [%(default)s].')
    m.add_argument('--layer_4_size', nargs='+', default=[4096], type=int,
                   help='List of layer (4) size to try [%(default)s].')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # exp_folder = os.path.join(args.out_folder, args.exp_name)
    in_folder = os.path.join(os.getcwd(), args.in_folder)
    in_labels = os.path.join(os.getcwd(), args.in_labels)

    if args.log_level == 'DEBUG':
        verbose = 3
        coloredlogs.install(level=args.log_level)
    elif args.log_level == 'INFO':
        verbose = 1
        coloredlogs.install(level=args.log_level)
    elif args.log_level == 'SILENT':
        verbose = 0
        coloredlogs.install(level='ERROR')

    config = {
            "l1": tune.grid_search(args.layer_1_size),
            "l2": tune.grid_search(args.layer_2_size),
            "l3": tune.grid_search(args.layer_3_size),
            "l4": tune.grid_search(args.layer_4_size),
        }

    apply_gan(args.in_models, args.in_folder, args.in_labels, args.include, args.exclude)
    shutdown()


if __name__ == "__main__":
    main()
