#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse
import os
from CoNNet.train import train_classification, test_classification, load_data
from ray import tune, init, shutdown
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from functools import partial
import shutil
import logging
import coloredlogs


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_folder',
                   help='')
    p.add_argument('in_labels',
                   help='')
    p.add_argument('--out_folder', default='./ray_results/',
                   help='')
    p.add_argument('--exp_name', default='classif_connectome',
                   help='')

    p.add_argument('--epoch', type=int, default=100,
                   help='')
    p.add_argument('--num_samples', type=int, default=1,
                   help='')
    p2 = p.add_mutually_exclusive_group()
    p2.add_argument('--resume', action='store_true',
                    help='')
    p2.add_argument('--overwrite', action='store_true',
                    help='')
    p.add_argument('--log_level', default='INFO',
                   help='')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    exp_folder = os.path.join(args.out_folder, args.exp_name)
    if os.path.isdir(exp_folder):
        if not args.overwrite and not args.resume:
            parser.error('Folder exists, use --overwrite or --resume.')
        elif args.overwrite:
            shutil.rmtree(exp_folder)
    if os.path.isdir(exp_folder) and not os.listdir(exp_folder) \
            and args.resume:
        args.resume = None

    if args.log_level == 'DEBUG':
        verbose = 3
        coloredlogs.install(level=args.log_level)
    elif args.log_level == 'INFO':
        verbose = 1
        coloredlogs.install(level=args.log_level)
    elif args.log_level == 'SILENT':
        verbose = 0
        coloredlogs.install(level='ERROR')

    in_folder = os.path.join(os.getcwd(), args.in_folder)
    in_labels = os.path.join(os.getcwd(), args.in_labels)

    init()
    config = {
        "l1": tune.choice([16, 24, 32]),
        "l2": tune.choice([32, 48, 64]),
        "l3": tune.choice([64, 128, 256]),
        "lr": tune.choice([0.005, 0.001]),
        "batch_size": tune.choice([25, 50])
    }
    # config = {
    #     "l1": tune.choice([32]),
    #     "l2": tune.choice([64]),
    #     "l3": tune.choice([256]),
    #     "lr": tune.choice([0.005]),
    #     "batch_size": tune.choice([25])
    # }

    reporter = CLIReporter(
        parameter_columns=["l1", "l2", 'l3', "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(train_classification,
                in_folder=in_folder,
                in_labels=in_labels,
                num_epoch=args.epoch),
        name=args.exp_name,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=args.num_samples,
        verbose=verbose,
        progress_reporter=reporter,
        resume=args.resume,
        local_dir=args.out_folder)

    test_classification(result, in_folder, in_labels)
    shutdown()


if __name__ == "__main__":
    main()
