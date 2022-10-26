#!/usr/bin/env python

from collections import OrderedDict
from genericpath import isdir
import itertools
import os
import random
import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import (ConcatDataset,
                              DataLoader,
                              SubsetRandomSampler)
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from ray import tune

from CoNNet.utils import (color_print, load_data, ConnectomeDataset,
                          add_noise, add_connections, remove_connections,
                          remove_row_column, balance_sampler,
                          get_n_params)
from CoNNet.sampler import WeightedRandomSampler
# from CoNNet.models import BrainNetCNN_double
from CoNNet.gan import BrainNetCNN_Discriminator, BrainNetCNN_Generator
# from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()
LOSS_LAMDA_1 = 0.1
LOSS_LAMDA_2 = 0.1
LOSS_IDENTITY = 0.5
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


def compute_scores(preds, ytrue_c):
    # TODO review type of input
    """
    Compute all scores for both classifications and regression tasks.

    Parameters
    ----------
    preds : list
        List of predictions for all tasks.
        Length of sum(nbr_classes_each) + nbr_regression
    ytrue_c : list
        List of target classes

    Returns
    -------
    tuple (4,)
        Scores for the entire set of tasks: acc, f1, 0, 0
    """
    preds = np.concatenate(preds)
    ytrue_c = np.concatenate(ytrue_c)

    acc, f1 = 0.0, 0.0
    print('--------------------')
    print(len(preds.argmax(axis=1)), len(ytrue_c))
    print(np.count_nonzero(preds.argmax(axis=1) == 1),
          np.count_nonzero(ytrue_c == 1))
    print('--------------------')
    acc += accuracy_score(preds.argmax(axis=1),  ytrue_c)
    f1 += f1_score(preds.argmax(axis=1), ytrue_c, average='weighted')

    return acc, f1, 0, 0


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
                             how_many=config['how_many'], cycle_GaN=True)

    # Number of features / matrix size
    nbr_features = len(loaded_stuff[-1])
    matrix_size = loaded_stuff[-2]
    nbr_classification = loaded_stuff[2].shape[-1] if not np.any(
        np.isnan(loaded_stuff[2])) else 0
    nbr_sites = len(np.unique(loaded_stuff[2][:, 0]))

    generators = OrderedDict()
    for i in range(nbr_sites)[1:]:
        generators['{}_to_{}'.format(0, i)] = BrainNetCNN_Generator(
            nbr_features, matrix_size)
        generators['{}_to_{}'.format(i, 0)] = BrainNetCNN_Generator(
            nbr_features, matrix_size)

    discriminators = [BrainNetCNN_Discriminator(nbr_features, matrix_size,
                                                l1=config['l1'], l2=config['l2'],
                                                l3=config['l3'], l4=config['l4'])
                      for _ in range(nbr_sites)]

    if checkpoint_dir:
        for key in generators.keys():
            filename = os.path.join(checkpoint_dir,
                                    'generator_{}'.format(key))
            if os.path.isfile(filename):
                generators[key].load_state_dict(torch.load(filename))

        for i in range(nbr_sites):
            filename = os.path.join(checkpoint_dir,
                                    'discriminator_{}'.format(i))
            if os.path.isfile(filename):
                discriminators[i].load_state_dict(torch.load(filename))

    if use_cuda:
        for generator in generators.values():
            generator.cuda()
        for discriminator in discriminators:
            discriminator.cuda()

    generators_optimizer = OrderedDict()
    for key in generators.keys():
        generators_optimizer[key] = torch.optim.Adam(
            generators[key].parameters(),
            lr=config['lr_g'], weight_decay=config['wd_g'],
            amsgrad=True, eps=1e-6)

    discriminators_optimizer = []
    for i in range(len(discriminators)):
        discriminators_optimizer.append(torch.optim.Adam(
            discriminators[i].parameters(),
            lr=config['lr_d'], weight_decay=config['wd_d'],
            amsgrad=True, eps=1e-6))

    CE_loss = nn.CrossEntropyLoss()
    c_loss_1 = nn.L1Loss()
    c_loss_2 = nn.MSELoss()

    # Prepare train/val with data augmentation
    transform = transforms.Compose([add_connections(), remove_connections()])
    trainset_ori = ConnectomeDataset(loaded_stuff, mode='train',
                                     transform=False, cycle_GaN=True)
    trainset_add_rem = ConnectomeDataset(loaded_stuff, mode='train',
                                         transform=transform, cycle_GaN=True)
    trainset_noise = ConnectomeDataset(loaded_stuff, mode='train',
                                       transform=add_noise(), cycle_GaN=True)
    trainset_row_col = ConnectomeDataset(loaded_stuff, mode='train',
                                         transform=remove_row_column(), cycle_GaN=True)
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
            # trainset[new_val[-1]]
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
            for generator in generators.values():
                generator.train()
            for discriminator in discriminators:
                discriminator.train()

            running_loss = 0.0
            running_cycle_loss = 0.0
            preds = []
            ytrue_c = []
            for batch_idx, (inputs, targets_c) in enumerate(trainloader):
                if use_cuda:
                    inputs, targets_c = inputs.cuda(), targets_c.cuda().long()

                split_inputs_true = []
                for i in range(nbr_sites):
                    idx = torch.squeeze(targets_c) == i
                    split_inputs_true.append(inputs[idx])

                all_losses = 0.0
                for key in generators.keys():
                    start, finish = key.split('_to_')
                    start, finish = int(start), int(finish)
                    if len(split_inputs_true[start]) == 0 or \
                            len(split_inputs_true[finish]) == 0:
                        continue

                    flip_key = '{}_to_{}'.format(finish, start)
                    generators_optimizer[key].zero_grad()
                    generators_optimizer[flip_key].zero_grad()

                    tmp_alt = generators[key](split_inputs_true[start])
                    tmp_reconst = generators[flip_key](tmp_alt)

                    idt_start = generators[flip_key](split_inputs_true[start])
                    idt_finish = generators[key](split_inputs_true[finish])

                    cycle_loss_1 = c_loss_1(
                        split_inputs_true[start], tmp_reconst) / LOSS_LAMDA_1
                    cycle_loss_2 = c_loss_2(
                        split_inputs_true[start], tmp_reconst) / LOSS_LAMDA_2

                    all_losses = cycle_loss_1 + cycle_loss_2
                    if LOSS_IDENTITY > 0:
                        idt_start_loss_1 = c_loss_1(
                            split_inputs_true[start], idt_start) / LOSS_LAMDA_1
                        idt_start_loss_2 = c_loss_2(
                            split_inputs_true[start], idt_start) / LOSS_LAMDA_2
                        all_losses = cycle_loss_1 + cycle_loss_2
                        idt_finish_loss_1 = c_loss_1(
                            split_inputs_true[finish], idt_finish) / LOSS_LAMDA_1
                        idt_finish_loss_2 = c_loss_2(
                            split_inputs_true[finish], idt_finish) / LOSS_LAMDA_2
                        all_losses += (idt_start_loss_1 + idt_start_loss_2 +
                                       idt_finish_loss_1 + idt_finish_loss_2)*LOSS_IDENTITY
                    running_loss += all_losses

                    all_losses.backward()
                    generators_optimizer[key].step()
                    generators_optimizer[flip_key].step()
                    running_cycle_loss += float(cycle_loss_1 + cycle_loss_2)

                    print('cycle_loss_1', float(cycle_loss_1))
                    print('cycle_loss_2', float(cycle_loss_2))
                    print('identity_loss_1', float(
                        idt_start_loss_1+idt_finish_loss_1))
                    print('identity_loss_2', float(
                        idt_start_loss_2+idt_finish_loss_2))

                    # Discriminator start
                    false_labels = torch.zeros(len(
                        tmp_alt)).cuda().long()
                    true_labels = torch.ones(len(
                        split_inputs_true[start])).cuda().long()
                    discriminators_optimizer[start].zero_grad()

                    origin_dis_out = discriminators[start](
                        tmp_reconst.detach())
                    origin_dis_loss = CE_loss(origin_dis_out,
                                              false_labels)

                    true_origin_dis_out = discriminators[start](
                        split_inputs_true[start])
                    true_origin_dis_loss = CE_loss(true_origin_dis_out,
                                                   true_labels)

                    all_losses = origin_dis_loss + true_origin_dis_loss
                    all_losses.backward()
                    running_loss += all_losses
                    discriminators_optimizer[start].step()

                    if use_cuda:
                        origin_dis_out, true_origin_dis_out = \
                            origin_dis_out.cpu(), true_origin_dis_out.cpu()
                        false_labels, true_labels = false_labels.cpu(), true_labels.cpu()
                    outputs = torch.cat(
                        [origin_dis_out, true_origin_dis_out]).detach().numpy()
                    targets_c = torch.cat(
                        [false_labels, true_labels]).detach().numpy()
                    preds.append(outputs)
                    ytrue_c.append(targets_c)
                    print('origin_dis_out_start', float(origin_dis_loss))
                    print('true_origin_dis_out_start',
                          float(true_origin_dis_loss))

                    # Discriminator finish
                    false_labels = torch.zeros(len(
                        tmp_alt)).cuda().long()
                    true_labels = torch.ones(len(
                        split_inputs_true[finish])).cuda().long()
                    discriminators_optimizer[finish].zero_grad()

                    target_dis_out = discriminators[finish](
                        tmp_alt.detach())
                    target_dis_loss = CE_loss(target_dis_out, false_labels)

                    true_target_dis_out = discriminators[finish](
                        split_inputs_true[finish])
                    true_target_dis_loss = CE_loss(true_target_dis_out,
                                                   true_labels)

                    all_losses = target_dis_loss + true_target_dis_loss
                    all_losses.backward()
                    running_loss += all_losses
                    discriminators_optimizer[finish].step()

                    if use_cuda:
                        target_dis_out, true_target_dis_out = \
                            target_dis_out.cpu(), true_target_dis_out.cpu()
                        false_labels, true_labels = false_labels.cpu(), true_labels.cpu()
                    outputs = torch.cat(
                        [target_dis_out, true_target_dis_out]).detach().numpy()
                    targets_c = torch.cat(
                        [false_labels, true_labels]).detach().numpy()
                    preds.append(outputs)
                    ytrue_c.append(targets_c)
                    print('target_dis_loss_finish', float(target_dis_loss))
                    print('true_target_dis_loss_finish',
                          float(true_target_dis_loss))

                    print()

            loss_train = running_loss/(batch_idx+1)
            cycle_loss = running_cycle_loss/(batch_idx+1)
            acc, f1, _, corr = compute_scores(preds, ytrue_c)

            writer.add_scalar('Loss/train', loss_train, epoch)
            writer.add_scalar('Accuracy/train', acc, epoch)
            writer.add_scalar('F1/train', f1, epoch)
            writer.add_scalar('cycle_loss/train', cycle_loss, epoch)
            writer.add_scalar('discriminators_loss/train',
                              loss_train-cycle_loss, epoch)
            color_print('loss:{}, acc:{}, f1:{}, cycle_loss:{}, discriminators_loss:{}'.format(
                loss_train, acc, f1, cycle_loss, loss_train-cycle_loss))

            # Validation phase
            generator.eval()
            discriminator.eval()
            running_loss = 0.0
            running_cycle_loss = 0.0
            preds = []
            ytrue_c = []
            ytrue_r = []
            for batch_idx, (inputs, targets_c) in enumerate(valloader):
                if use_cuda:
                    inputs, targets_c = inputs.cuda(), targets_c.cuda().long()

                with torch.no_grad():
                    split_inputs_true = []
                    for i in range(nbr_sites):
                        idx = torch.squeeze(targets_c) == i
                        split_inputs_true.append(inputs[idx])

                    if len(split_inputs_true[start]) == 0 or \
                            len(split_inputs_true[finish]) == 0:
                        continue

                    # DEBUG #
                    if batch_idx == 0:
                        test_mat_a = torch.unsqueeze(
                            trainset[247][0], dim=0).cuda()
                        print(test_mat_a.shape)
                        test_mat_b = torch.unsqueeze(
                            trainset[152][0], dim=0).cuda()
                        start, finish = 0, 1
                        key = '{}_to_{}'.format(start, finish)
                        flip_key = '{}_to_{}'.format(finish, start)

                        print('a')
                        tmp_alt = generators[key](test_mat_a)
                        print('b')
                        tmp_reconst = generators[flip_key](tmp_alt)
                        print('c')
                        path = '/home/frheault/Datasets/CoNNet/debug/'
                        ori = test_mat_a.detach().cpu().numpy()
                        reconst = tmp_reconst.detach().cpu().numpy()
                        np.save(path+'ori_a_{}'.format(epoch), ori)
                        np.save(path+'reconst_a_{}'.format(epoch), reconst)

                        tmp_alt = generators[flip_key](test_mat_b)
                        tmp_reconst = generators[key](tmp_alt)
                        path = '/home/frheault/Datasets/CoNNet/debug/'
                        ori = test_mat_b.detach().cpu().numpy()
                        reconst = tmp_reconst.detach().cpu().numpy()
                        np.save(path+'ori_b_{}'.format(epoch), ori)
                        np.save(path+'reconst_b_{}'.format(epoch), reconst)

                    all_losses = 0.0
                    for key in generators.keys():
                        start, finish = key.split('_to_')
                        start, finish = int(start), int(finish)
                        flip_key = '{}_to_{}'.format(finish, start)

                        tmp_alt = generators[key](
                            split_inputs_true[start])
                        tmp_reconst = generators[flip_key](tmp_alt)
                        idt_start = generators[flip_key](
                            split_inputs_true[start])
                        idt_finish = generators[key](split_inputs_true[finish])

                        false_labels = torch.zeros(len(tmp_alt)).cuda().long()
                        true_labels = torch.ones(
                            len(split_inputs_true[finish])).cuda().long()

                        cycle_loss_1 = c_loss_1(
                            split_inputs_true[start], tmp_reconst) / LOSS_LAMDA_1
                        cycle_loss_2 = c_loss_2(
                            split_inputs_true[start], tmp_reconst) / LOSS_LAMDA_2

                        all_losses = cycle_loss_1 + cycle_loss_2
                        if LOSS_IDENTITY > 0:
                            idt_start_loss_1 = c_loss_1(
                                split_inputs_true[start], idt_start) / LOSS_LAMDA_1
                            idt_start_loss_2 = c_loss_2(
                                split_inputs_true[start], idt_start) / LOSS_LAMDA_2
                            all_losses = cycle_loss_1 + cycle_loss_2
                            idt_finish_loss_1 = c_loss_1(
                                split_inputs_true[finish], idt_finish) / LOSS_LAMDA_1
                            idt_finish_loss_2 = c_loss_2(
                                split_inputs_true[finish], idt_finish) / LOSS_LAMDA_2
                            all_losses += (idt_start_loss_1 + idt_start_loss_2 +
                                           idt_finish_loss_1 + idt_finish_loss_2)*LOSS_IDENTITY

                        # Discriminator start
                        false_labels = torch.zeros(
                            len(tmp_reconst)).cuda().long()
                        true_labels = torch.ones(
                            len(split_inputs_true[start])).cuda().long()
                        origin_dis_out = discriminators[start](
                            tmp_reconst.detach())
                        origin_dis_loss = CE_loss(origin_dis_out, false_labels)

                        true_origin_dis_out = discriminators[start](
                            split_inputs_true[start])
                        true_origin_dis_loss = CE_loss(
                            true_origin_dis_out, true_labels)

                        if use_cuda:
                            origin_dis_out, true_origin_dis_out = \
                                origin_dis_out.cpu(), true_origin_dis_out.cpu()
                            false_labels, true_labels = false_labels.cpu(), true_labels.cpu()
                        outputs = torch.cat(
                            [origin_dis_out, true_origin_dis_out]).detach().numpy()
                        targets_c = torch.cat(
                            [false_labels, true_labels]).detach().numpy()
                        preds.append(outputs)
                        ytrue_c.append(targets_c)

                        # Discriminator finish
                        false_labels = torch.zeros(len(tmp_alt)).cuda().long()
                        true_labels = torch.ones(
                            len(split_inputs_true[finish])).cuda().long()
                        target_dis_out = discriminators[finish](
                            tmp_alt.detach())
                        target_dis_loss = CE_loss(target_dis_out, false_labels)

                        true_target_dis_out = discriminators[finish](
                            split_inputs_true[finish])
                        true_target_dis_loss = CE_loss(
                            true_target_dis_out, true_labels)

                        all_losses = origin_dis_loss + true_origin_dis_loss + \
                            target_dis_loss + true_target_dis_loss + \
                            cycle_loss_1 + cycle_loss_2
                        running_loss += all_losses
                        running_cycle_loss += cycle_loss_1 + cycle_loss_2

                        if use_cuda:
                            target_dis_out, true_target_dis_out = \
                                target_dis_out.cpu(), true_target_dis_out.cpu()
                            false_labels, true_labels = false_labels.cpu(), true_labels.cpu()
                        outputs = torch.cat(
                            [target_dis_out, true_target_dis_out]).detach().numpy()
                        targets_c = torch.cat(
                            [false_labels, true_labels]).detach().numpy()
                        preds.append(outputs)
                        ytrue_c.append(targets_c)

            loss_val = running_loss/(batch_idx+1)
            cycle_loss = running_cycle_loss/(batch_idx+1)
            acc, f1, _, _ = compute_scores(preds, ytrue_c)

            writer.add_scalar('Loss/val', loss_val, epoch)
            writer.add_scalar('Accuracy/val', acc, epoch)
            writer.add_scalar('F1/val', f1, epoch)
            writer.add_scalar('cycle_loss/val', cycle_loss, epoch)
            writer.add_scalar('discriminators_loss/val',
                              loss_val-cycle_loss, epoch)

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                for key in generators.keys():
                    filename = os.path.join(
                        checkpoint_dir, 'generator_{}'.format(key))
                    torch.save(generators[key].state_dict(), filename)

                for i in range(nbr_sites):
                    filename = os.path.join(
                        checkpoint_dir, 'discriminator_{}'.format(i))
                    torch.save(discriminators[i].state_dict(),
                               checkpoint_dir + '/discriminator_{}'.format(i))

            tune.report(loss=loss_val, cycle_loss=cycle_loss,
                        discriminators_loss=loss_val-cycle_loss,
                        accuracy=acc, f1_score=f1)


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
    CE_loss = nn.CrossEntropyLoss()
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
        discriminator = BrainNetCNN_Discriminator(nbr_features, matrix_size,
                                                  num_class=nbr_classes_each[0]+1,
                                                  l1=best_trial.config['l1'],
                                                  l2=best_trial.config['l2'],
                                                  l3=best_trial.config['l3'],
                                                  l4=best_trial.config['l4'])

        if use_cuda:
            discriminator = discriminator.cuda(0)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, _ = torch.load(os.path.join(best_checkpoint_dir,
                                                 "checkpoint"))
        discriminator.load_state_dict(model_state)

    if save_best_model is not None:
        torch.save(discriminator, save_best_model)

    running_loss = 0.0
    preds = []
    ytrue_c = []
    ytrue_r = []
    for batch_idx, (inputs, tabs, targets_c, targets_r) in enumerate(testloader):
        if use_cuda:
            inputs, tabs, targets_c, targets_r = inputs.cuda(), \
                tabs.cuda(), targets_c.cuda().long(), targets_r.cuda()

        with torch.no_grad():
            tabs = None if nbr_tabular == 0 else tabs
            outputs = discriminator(inputs)
            loss = CE_loss(outputs, torch.squeeze(targets_c))

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


def apply_gan(in_models, in_folder, in_labels, filenames_to_include, filenames_to_exclude, final_target='0', out_dir='./apply/'):
    """
    Apply the GAN to a dataset
    """
    set_seed()
    if filenames_to_exclude is None:
        filenames_to_exclude = ['tot_commit2_weights.npy',
                                'sc_edge_normalized.npy',
                                'sc_vol_normalized.npy']

    loaded_stuff = load_data(directory_path=in_folder,
                             labels_path=in_labels,
                             features_filename_include=filenames_to_include,
                             features_filename_exclude=filenames_to_exclude,
                             how_many=10000000, cycle_GaN=True)
    # Number of features / matrix size
    nbr_features = len(loaded_stuff[-1])
    matrix_size = loaded_stuff[-2]

    # Number of features / matrix size
    dict_model = {}
    for filename in in_models:
        basename = os.path.basename(filename).replace('generator_', '')
        source, target = basename.split('_to_')
        net = BrainNetCNN_Generator(nbr_features, matrix_size)
        net.load_state_dict(torch.load(filename))
        if target == final_target:
            dict_model[(source, final_target)] = net

    dataset_ori = ConnectomeDataset(loaded_stuff, mode='both',
                                     transform=False, cycle_GaN=True)

    for i in range(len(dataset_ori)):
        if str(int(dataset_ori.Y_c[i])) != final_target:
            curr_model = dict_model[(str(int(dataset_ori.Y_c[i])), final_target)]
            reconst = curr_model(torch.unsqueeze(dataset_ori[i][0], dim=0))
            basedir = os.path.basename(dataset_ori.filepaths[i])
            tmp_dir = os.path.join(out_dir, basedir)

            if not os.path.isdir(tmp_dir):
                os.mkdir(tmp_dir)
            for i in range(len(dataset_ori.features_filename)):
                tmp_name = os.path.join(tmp_dir, dataset_ori.features_filename[i])
                np.save(tmp_name, reconst[0, i, :, :].detach().numpy())

