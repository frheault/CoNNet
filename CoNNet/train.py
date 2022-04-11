#!/usr/bin/env python

import os
import random
import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import torch
from torch.utils.data import (ConcatDataset,
                              DataLoader,
                              SubsetRandomSampler)
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from ray import tune

from CoNNet.utils import (color_print, load_data, ConnectomeDataset,
                          add_noise, add_connections, remove_connections,
                          remove_row_column, balance_sampler)
from CoNNet.sampler import WeightedRandomSampler
from CoNNet.models import BrainNetCNN_double
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()

# TODO triple-check it is deterministic


def set_seed():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def customize_loss(nbr_classification, nbr_classes_each, nbr_regression,
                   outputs, targets_c, targets_r):
    # TODO review type of input
    """
    Compute all losses for both classifications and regression tasks.

    Parameters
    ----------
    nbr_classification : int
       Number of classification tasks.
    nbr_classes_each : list (int)
        List of length classification for the size of the target space of
        each classification task.
    nbr_regression : int
        Number of regression tasks.
    outputs : list
        List of predictions for all tasks.
        Length of sum(nbr_classes_each) + nbr_regression
    targets_c : list
        List of target classes
    targets_r : list
        List of target values

    Returns
    -------
    torch.nn.Loss
        Sum of losses from all tasks
    """
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_MSE = torch.nn.MSELoss()

    loss_c = 0.0
    loss_r = 0.0
    skip = 0
    for i in range(nbr_classification):
        loss_c += criterion_CE(outputs[:, skip:skip +
                                       nbr_classes_each[i]], targets_c[:, i])
        skip += nbr_classes_each[i]

    for i in range(nbr_regression):
        loss_r += criterion_MSE(outputs[:, skip+i], targets_r[:, i])
    return loss_c + loss_r


def compute_scores(nbr_classification, nbr_classes_each,
                   nbr_regression, preds, ytrue_c, ytrue_r):
    # TODO review type of input
    """
    Compute all scores for both classifications and regression tasks.

    Parameters
    ----------
    nbr_classification : int
       Number of classification tasks.
    nbr_classes_each : list (int)
        List of length classification for the size of the target space of
        each classification task.
    nbr_regression : int
        Number of regression tasks.
    preds : list
        List of predictions for all tasks.
        Length of sum(nbr_classes_each) + nbr_regression
    ytrue_c : list
        List of target classes
    ytrue_r : list
        List of target values

    Returns
    -------
    tuple (4,)
        Scores for the entire set of tasks: acc, f1, maer, corr
    """
    preds = np.concatenate(preds)
    ytrue_c = np.concatenate(ytrue_c)
    ytrue_r = np.concatenate(ytrue_r)

    maer, corr = 0.0, 0.0
    acc, f1 = 0.0, 0.0

    skip = 0
    for i in range(nbr_classification):
        acc += accuracy_score(preds[:, skip:skip +
                                    nbr_classes_each[i]].argmax(axis=1),
                              np.hstack(ytrue_c[:, i]))
        f1 += f1_score(preds[:, skip:skip +
                             nbr_classes_each[i]].argmax(axis=1),
                       np.hstack(ytrue_c[:, i]), average='weighted')
        skip += nbr_classes_each[i]

    for i in range(nbr_regression):
        maer += mae(preds[:, skip+i], ytrue_r[:, i])
        corr += pearsonr(preds[:, skip+i], ytrue_r[:, i])[0]

    acc = acc/nbr_classification if nbr_classification > 0 else acc
    f1 = f1/nbr_classification if nbr_classification > 0 else f1
    maer = maer/nbr_regression if nbr_regression > 0 else maer
    corr = corr/nbr_regression if nbr_regression > 0 else corr

    return acc, f1, maer, corr


def train_classification(config, in_folder=None, in_labels=None, num_epoch=100,
                         nb_fold=False, adaptive_lr=None, balance_class=False,
                         checkpoint_dir=None, pre_training=None,
                         filenames_to_include=None,
                         filenames_to_exclude=None):
    # TODO docstring
    set_seed()
    if filenames_to_exclude is None:
        filenames_to_exclude = ['tot_commit2_weights.npy',
                                'sc_edge_normalized.npy',
                                'sc_vol_normalized.npy']

    loaded_stuff = load_data(directory_path=in_folder,
                             labels_path=in_labels,
                             features_filename_include=filenames_to_include,
                             features_filename_exclude=filenames_to_exclude,
                             how_many=config['how_many'])

    # Number of features / matrix size
    nbr_features = len(loaded_stuff[-1])
    matrix_size = loaded_stuff[-2]
    nbr_classification = loaded_stuff[2].shape[-1] if not np.any(
        np.isnan(loaded_stuff[2])) else 0
    nbr_classes_each = [len(np.unique(loaded_stuff[2][:, i]))
                        for i in range(nbr_classification)]
    nbr_regression = len(loaded_stuff[3][0]) if not np.any(
        np.isnan(loaded_stuff[3])) else 0
    nbr_tabular = len(loaded_stuff[4][0]) if not np.any(
        np.isnan(loaded_stuff[4])) else 0

    if pre_training:
        if os.path.isfile(pre_training):
            net_pre = torch.load(pre_training)
            net = BrainNetCNN_double(nbr_features, matrix_size,
                                     nbr_classification, nbr_classes_each,
                                     nbr_regression, nbr_tabular,
                                     l1=config['l1'], l2=config['l2'],
                                     l3=config['l3'], l4=config['l4'])

            logging.info('Using pre-training! Transfering weigths')
            net.E2Econv1.cnn1.weight = net_pre.E2Econv1.cnn1.weight
            net.E2Econv1.cnn2.weight = net_pre.E2Econv1.cnn2.weight
            net.E2Econv2.cnn1.weight = net_pre.E2Econv2.cnn1.weight
            net.E2Econv2.cnn2.weight = net_pre.E2Econv2.cnn2.weight
            net.E2N.weight = net_pre.E2N.weight
            net.N2G.weight = net_pre.N2G.weight

            logging.info('Using pre-training! Freezing layers.')
            net.E2Econv1.cnn1.weight.requires_grad = False
            net.E2Econv1.cnn2.weight.requires_grad = False
            net.E2Econv2.cnn1.weight.requires_grad = False
            net.E2Econv2.cnn2.weight.requires_grad = False
            net.E2N.weight.requires_grad = False
            net.N2G.weight.requires_grad = False
        else:
            logging.warning('Pre-training file does not exist!\n'
                            'Pre-training NOT activated.')
    else:
        net = BrainNetCNN_double(nbr_features, matrix_size,
                                 nbr_classification, nbr_classes_each,
                                 nbr_regression, nbr_tabular,
                                 l1=config['l1'], l2=config['l2'],
                                 l3=config['l3'], l4=config['l4'])

    if use_cuda:
        net = net.cuda(0)

    lr = config['lr']
    wd = config['wd']
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd,
                                 amsgrad=True, eps=1e-6)
    if adaptive_lr is not None:
        scheduler = StepLR(optimizer, step_size=adaptive_lr, gamma=0.5)

    if checkpoint_dir:
        filename = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.isfile(filename):
            model_state, optimizer_state = torch.load(filename)
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    # Prepare train/val with data augmentation
    transform = transforms.Compose([add_connections(), remove_connections()])
    trainset_ori = ConnectomeDataset(loaded_stuff, mode='train',
                                     transform=False)
    trainset_add_rem = ConnectomeDataset(loaded_stuff, mode='train',
                                         transform=transform)
    trainset_noise = ConnectomeDataset(loaded_stuff, mode='train',
                                       transform=add_noise())
    trainset_row_col = ConnectomeDataset(loaded_stuff, mode='train',
                                         transform=remove_row_column())
    trainset = ConcatDataset([trainset_ori, trainset_add_rem,
                              trainset_noise, trainset_row_col])

    # Split training set in two (train/validation)
    # The data has 3 types of data augmentation, appended one after the
    # other. To avoid data contamination between train/validation data
    # is picked from the original set and its associated duplicate selected
    max_len = len(trainset) // 4
    all_idx = np.arange(max_len)
    random.shuffle(all_idx)

    if nb_fold:
        kf_split = KFold(n_splits=nb_fold, shuffle=False, random_state=None)
        split_method = kf_split.split(all_idx)
    else:
        split_pos = int(len(all_idx) * 0.8)
        split_method = [[all_idx[:split_pos], all_idx[split_pos:]]]

    for fold, (train_idx, val_idx) in enumerate(split_method):
        color_print('Fold #{}, Datasets (with augmentation): '
                    '{} train and {} val'.format(
                        fold, len(train_idx), len(val_idx)))

        # Considering the split train/validation, pick the associated
        # duplicated datasets in each of the data augmentation
        # 1-2-3-1a-2a-3a-1b-2b-3c-1c-2c-3c
        # 1-2 = train -> 1-2-1a-2a-1b-2b-1c-2c
        # 3 = train -> 3-3a-3b-3c
        real_train_idx = []
        real_val_idx = []
        for i in range(4):
            new_train = np.array(train_idx, dtype=int) + (i * max_len)
            real_train_idx.extend(new_train)
            new_val = np.array(val_idx, dtype=int) + (i * max_len)
            real_val_idx.extend(new_val)
            trainset[new_val[-1]]
        train_sampler = SubsetRandomSampler(real_train_idx)
        val_sampler = SubsetRandomSampler(real_val_idx)

        # Only works if doing a single classification task!
        if balance_class:
            rng_cpu = torch.Generator()
            rng_cpu.manual_seed(1066)
            class_w = balance_sampler(trainset, real_train_idx)
            train_sampler = WeightedRandomSampler(real_train_idx, class_w,
                                                  generator=rng_cpu)
            class_w = balance_sampler(trainset, real_val_idx)
            val_sampler = WeightedRandomSampler(real_val_idx, weights=class_w,
                                                generator=rng_cpu)
        else:
            train_sampler = SubsetRandomSampler(real_train_idx)
            val_sampler = SubsetRandomSampler(real_val_idx)

        set_seed()
        trainloader = DataLoader(trainset, batch_size=int(config['batch_size']),
                                 num_workers=1, sampler=train_sampler,
                                 worker_init_fn=seed_worker)

        valloader = DataLoader(trainset, batch_size=int(config['batch_size']),
                               num_workers=1, sampler=val_sampler,
                               worker_init_fn=seed_worker)
        writer = SummaryWriter()
        for epoch in range(num_epoch):
            net.train()
            running_loss = 0.0
            preds = []
            ytrue_c = []
            ytrue_r = []
            for batch_idx, (inputs, tabs, targets_c, targets_r) in enumerate(trainloader):
                if use_cuda:
                    inputs, tabs, targets_c, targets_r = inputs.cuda(), \
                        tabs.cuda(), targets_c.cuda().long(), targets_r.cuda()
                optimizer.zero_grad()

                tabs = None if nbr_tabular == 0 else tabs
                outputs = net(inputs, tabs)

                loss = customize_loss(nbr_classification, nbr_classes_each,
                                      nbr_regression, outputs,
                                      targets_c, targets_r)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data.item()
                if use_cuda:
                    outputs, targets_c, targets_r = outputs.cpu(), \
                        targets_c.cpu(), targets_r.cpu()

                preds.append(outputs.detach().numpy())
                ytrue_c.append(targets_c.detach().numpy())
                ytrue_r.append(targets_r.detach().numpy())

                if adaptive_lr is not None:
                    scheduler.step()

            loss_train = running_loss/(batch_idx+1)
            acc, f1, maer, corr = compute_scores(nbr_classification,
                                                 nbr_classes_each,
                                                 nbr_regression,
                                                 preds, ytrue_c, ytrue_r)

            writer.add_scalar('Loss/train', loss_train, epoch)
            writer.add_scalar('Accuracy/train', acc, epoch)
            writer.add_scalar('F1/train', f1, epoch)
            writer.add_scalar('MAE/train', maer, epoch)
            writer.add_scalar('CORR/train', corr, epoch)
            color_print('loss:{}, acc:{}, f1:{}, mae:{}, corr:{}'.format(
                loss_train, acc, f1, maer, corr))

            # Validation phase
            net.eval()
            running_loss = 0.0
            preds = []
            ytrue_c = []
            ytrue_r = []
            for batch_idx, (inputs, tabs, targets_c, targets_r) in enumerate(valloader):
                if use_cuda:
                    inputs, tabs, targets_c, targets_r = inputs.cuda(), \
                        tabs.cuda(), targets_c.cuda().long(), targets_r.cuda()
                with torch.no_grad():
                    tabs = None if nbr_tabular == 0 else tabs
                    outputs = net(inputs, tabs)
                    loss = customize_loss(nbr_classification, nbr_classes_each,
                                          nbr_regression, outputs,
                                          targets_c, targets_r)

                    if use_cuda:
                        outputs, targets_c, targets_r = outputs.cpu(), \
                            targets_c.cpu(), targets_r.cpu()

                    preds.append(outputs.detach().numpy())
                    ytrue_c.append(targets_c.detach().numpy())
                    ytrue_r.append(targets_r.detach().numpy())

                running_loss += loss.data.item()

            # print('val',running_loss, batch_idx)
            loss_val = running_loss/(batch_idx+1)
            acc, f1, maer, corr = compute_scores(nbr_classification,
                                                 nbr_classes_each,
                                                 nbr_regression,
                                                 preds, ytrue_c, ytrue_r)

            writer.add_scalar('Loss/val', loss_val, epoch)
            writer.add_scalar('Accuracy/val', acc, epoch)
            writer.add_scalar('F1/val', f1, epoch)
            writer.add_scalar('MAE/val', maer, epoch)
            writer.add_scalar('CORR/val', corr, epoch)

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=loss_val, mae=maer, corr=corr, accuracy=acc,
                        f1_score=f1)


def test_classification(input_results, in_folder, in_labels,
                        filenames_to_include=None, filenames_to_exclude=None,
                        balance_class=False, save_best_model=None):
    if filenames_to_exclude is None:
        filenames_to_exclude = ['tot_commit2_weights.npy',
                                'sc_edge_normalized.npy',
                                'sc_vol_normalized.npy']
    loaded_stuff = load_data(directory_path=in_folder,
                             labels_path=in_labels,
                             features_filename_include=filenames_to_include,
                             features_filename_exclude=filenames_to_exclude,
                             how_many=100000)

    # Handle separately the test set
    testset_ori = ConnectomeDataset(loaded_stuff, mode='test',
                                    transform=False)
    testset = ConcatDataset([testset_ori])

    rng_cpu = torch.Generator()
    rng_cpu.manual_seed(2277)
    test_idx = list(range(len(testset)))
    if balance_class:
        class_w = balance_sampler(testset, test_idx)
        test_sampler = WeightedRandomSampler(test_idx, class_w,
                                             generator=rng_cpu)
    else:
        test_sampler = SubsetRandomSampler(test_idx)

    set_seed()
    testloader = DataLoader(testset, batch_size=50,
                            num_workers=1, sampler=test_sampler,
                            worker_init_fn=seed_worker)

    # All information about the network must be known in order to either load
    # it or to rebuild it (evaluation only, pre-training)
    nbr_features = len(loaded_stuff[-1])
    matrix_size = loaded_stuff[-2]
    nbr_classification = loaded_stuff[2].shape[-1] if not np.any(
        np.isnan(loaded_stuff[2])) else 0
    nbr_classes_each = [len(np.unique(loaded_stuff[2][:, i]))
                        for i in range(nbr_classification)]
    nbr_regression = len(loaded_stuff[3][0]) if not np.any(
        np.isnan(loaded_stuff[3])) else 0
    nbr_tabular = len(loaded_stuff[4][0]) if not np.any(
        np.isnan(loaded_stuff[4])) else 0
    if isinstance(input_results, str):
        net = torch.load(input_results)
        if use_cuda:
            net = net.cuda(0)
    else:
        best_trial = input_results.get_best_trial("loss", "min", "last")
        # Number of features / matrix size
        net = BrainNetCNN_double(nbr_features, matrix_size,
                                 nbr_classification, nbr_classes_each,
                                 nbr_regression, nbr_tabular,
                                 l1=best_trial.config['l1'],
                                 l2=best_trial.config['l2'],
                                 l3=best_trial.config['l3'],
                                 l4=best_trial.config['l4'])

        if use_cuda:
            net = net.cuda(0)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, _ = torch.load(os.path.join(best_checkpoint_dir,
                                                 "checkpoint"))
        net.load_state_dict(model_state)

    if save_best_model is not None:
        torch.save(net, save_best_model)

    running_loss = 0.0
    preds = []
    ytrue_c = []
    ytrue_r = []
    for batch_idx, (inputs, tabs, targets_c, targets_r) in enumerate(testloader):
        if use_cuda:
            inputs, tabs, targets_c, targets_r = inputs.cuda(), tabs.cuda(), \
                targets_c.cuda().long(), targets_r.cuda()

        with torch.no_grad():
            tabs = None if nbr_tabular == 0 else tabs
            outputs = net(inputs, tabs)
            loss = customize_loss(nbr_classification, nbr_classes_each,
                                  nbr_regression, outputs,
                                  targets_c, targets_r)

            running_loss += loss.data.item()

            if use_cuda:
                outputs, targets_c, targets_r = outputs.cpu(), \
                    targets_c.cpu(), targets_r.cpu()

            preds.append(outputs.detach().numpy())
            ytrue_c.append(targets_c.detach().numpy())
            ytrue_r.append(targets_r.detach().numpy())

    test_loss = running_loss/(batch_idx+1)
    acc, f1, maer, corr = compute_scores(nbr_classification,
                                         nbr_classes_each,
                                         nbr_regression,
                                         preds, ytrue_c, ytrue_r)

    color_print('TESTSET - loss:{}, acc:{}, f1:{}, mae:{}, corr:{}'.format(
        test_loss, acc, f1, maer, corr))

    if not isinstance(input_results, str):
        print(best_trial.config)
