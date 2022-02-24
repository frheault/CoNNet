#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse
from functools import partial
import os
import shutil

import coloredlogs
from ray import tune, init, shutdown

from CoNNet.train import (train_classification,
                          test_classification)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_folder',
                   help='Path to subjects connectivity matrices.')
    p.add_argument('in_labels',
                   help='XLSX file with subjects informations/labels')

    p.add_argument('--log_level', default='INFO',
                   help='Print extra logging information [%(default)s].')

    e = p.add_argument_group('Experiment options')
    e.add_argument('--out_folder', default='./ray_results/',
                   help='output folder for network checkpoints [%(default)s].')
    e.add_argument('--exp_name', default='classif_connectome',
                   help='Name of the current experiment [%(default)s].')
    e.add_argument('--pre_training', metavar='FILE',
                   help='Path to a model file for Torch to use as pre-training.')
    e1 = e.add_mutually_exclusive_group()
    e1.add_argument('--evaluate_test_set', metavar='FILE',
                    help='Skip training and only apply the network to test set.\n'
                         'User must provide a model file.')
    e1.add_argument('--save_best_model',
                    help='Save output model file (to reload or use as pre-training).')
    e2 = e.add_mutually_exclusive_group()
    e2.add_argument('--resume', action='store_true',
                    help='Resume experiment with --exp_name, checkpoints must '
                         'exist.')
    e2.add_argument('--overwrite', action='store_true',
                    help='Delete and restart experiment with --exp_name')

    t = p.add_argument_group('Training options')
    t.add_argument('--epoch', type=int, default=100,
                   help='Number of epoch for training.')
    t.add_argument('--k_fold', type=int, default=0,
                   help='Number of fold for training [%(default)s].')
    t.add_argument('--adaptive_lr', type=int, default=None,
                   help='Number of Epoch between halving')
    t.add_argument('--learning_rate', nargs='+', default=[0.0001], type=float,
                   help='List of learning rate  to try [%(default)s].')
    t.add_argument('--weight_decay', nargs='+', default=[0.005], type=float,
                   help='List of learning rate  to try [%(default)s].')
    t.add_argument('--batch_size', nargs='+', default=[50], type=int,
                   help='List of batch size to try [%(default)s].')
    t.add_argument('--limit_sample_size', nargs='+', default=None, type=int,
                   help='List of sample size size to try [%(default)s].')
    t2 = t.add_mutually_exclusive_group()
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

    exp_folder = os.path.join(args.out_folder, args.exp_name)
    in_folder = os.path.join(os.getcwd(), args.in_folder)
    in_labels = os.path.join(os.getcwd(), args.in_labels)

    if not args.evaluate_test_set:
        if args.resume and args.pre_training:
            args.pre_training = None

        # if os.path.isdir(exp_folder) and args.pre_training and not args.overwrite:
        #     parser.error('New experiment name must be defined if using '
        #                  '--pre_training.')
        if os.path.isdir(exp_folder):
            if not args.overwrite and not args.resume:
                parser.error('Folder exists, use --overwrite or --resume.')
            elif args.overwrite:
                shutil.rmtree(exp_folder)
        if not os.path.isdir(exp_folder) or not os.listdir(exp_folder) \
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

        init()
        if args.limit_sample_size is None:
            args.limit_sample_size = [1e8]
        args.pre_training = os.path.abspath(args.pre_training) \
            if args.pre_training is not None else None

        config = {
            "l1": tune.grid_search(args.layer_1_size),
            "l2": tune.grid_search(args.layer_2_size),
            "l3": tune.grid_search(args.layer_3_size),
            "l4": tune.grid_search(args.layer_4_size),
            "lr": tune.grid_search(args.learning_rate),
            "batch_size": tune.grid_search(args.batch_size),
            "wd": tune.grid_search(args.weight_decay),
            "how_many": tune.grid_search(args.limit_sample_size)
        }

        reporter = tune.CLIReporter(parameter_columns=["l1", "l2", 'l3', "lr"],
                                    metric_columns=["loss", "accuracy", "f1_score",
                                                    "mae", "corr", "training_iteration"])
        print(args)
        result = tune.run(
            partial(train_classification,
                    in_folder=in_folder,
                    in_labels=in_labels,
                    num_epoch=args.epoch,
                    nb_fold=args.k_fold,
                    adaptive_lr=args.adaptive_lr,
                    filenames_to_include=args.include,
                    filenames_to_exclude=args.exclude,
                    pre_training=args.pre_training),
            name=args.exp_name,
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=config,
            num_samples=1,
            verbose=verbose,
            progress_reporter=reporter,
            checkpoint_score_attr="min-loss",
            keep_checkpoints_num=10,
            # stop=stopper,
            resume=args.resume,
            local_dir=args.out_folder)

    if args.evaluate_test_set is not None and \
            not os.path.isfile(args.evaluate_test_set):
        parser.error('{} does not exist, cannot load model'.format(exp_folder))

    if args.evaluate_test_set and args.save_best_model:
        parser.error('Cannot use options to load a model and save best model')

    if args.evaluate_test_set:
        test_classification(args.evaluate_test_set, in_folder, in_labels,
                            filenames_to_include=args.include,
                            filenames_to_exclude=args.exclude)
    else:
        test_classification(result, in_folder, in_labels,
                            filenames_to_include=args.include,
                            filenames_to_exclude=args.exclude,
                            save_best_model=args.save_best_model)
    shutdown()


if __name__ == "__main__":
    main()
