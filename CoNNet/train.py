#!/usr/bin/env python

import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
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


def train_classification(config, in_folder=None, in_labels=None, num_epoch=1,
                         adaptative_lr=None,
                         checkpoint_dir=None, filenames_to_include=None,
                         filenames_to_exclude=None):
    set_seed()
    if filenames_to_exclude is None:
        filenames_to_exclude = ['tot_commit2_weights.npy', 'sc_edge_normalized.npy',
                                'sc_vol_normalized.npy']

    loaded_stuff = load_data(directory_path=in_folder,
                             labels_path=in_labels,
                             features_filename_include=filenames_to_include,
                             features_filename_exclude=filenames_to_exclude, how_many=config['how_many'])

    # Number of features / matrix size
    nbr_features = len(loaded_stuff[-1])
    matrix_size = loaded_stuff[-2]
    nbr_class = len(np.unique(loaded_stuff[1]))
    nbr_tabular = len(loaded_stuff[2][0]) if not np.any(
        np.isnan(loaded_stuff[2])) else 0

    net = BrainNetCNN_double(nbr_features, matrix_size, nbr_class, nbr_tabular,
                             l1=config['l1'], l2=config['l2'], l3=config['l3'])

    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])

    momentum = 0.9
    lr = config['lr']
    wd = config['wd']

    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    # Adam
    # optimizer = torch.optim.SGD(net.parameters(),
    #                             momentum=momentum,
    #                             lr=lr, weight_decay=wd, nesterov=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd,
                                 amsgrad=True, eps=1e-6)
    if adaptative_lr is not None:
        scheduler = StepLR(optimizer, step_size=adaptative_lr, gamma=0.5)

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
    max_len = len(trainset) // 4
    all_idx = np.arange(max_len)
    random.shuffle(all_idx)
    nb_fold = False
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

        real_train_idx = []
        real_val_idx = []
        # print(train_idx)
        # print(val_idx)
        for i in range(4):
            new_train = np.array(train_idx, dtype=int) + (i * max_len)
            real_train_idx.extend(new_train)
            new_val = np.array(val_idx, dtype=int) + (i * max_len)
            real_val_idx.extend(new_val)
            trainset[new_val[-1]]
        # train_sampler = SubsetRandomSampler(real_train_idx)
        # val_sampler = SubsetRandomSampler(real_val_idx)

        # print(val_sampler)
        rng_cpu = torch.Generator()
        rng_cpu.manual_seed(1066)
        class_w = balance_sampler(trainset, real_train_idx)
        train_sampler = WeightedRandomSampler(real_train_idx, class_w,
                                              generator=rng_cpu)
        class_w = balance_sampler(trainset, real_val_idx)
        val_sampler = WeightedRandomSampler(real_val_idx, weights=class_w,
                                            generator=rng_cpu)
        # print(real_train_idx)
        # print(real_val_idx)

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
            ytrue = []
            for batch_idx, (inputs, tabs, targets) in enumerate(trainloader):
                if use_cuda:
                    inputs, tabs, targets = inputs.cuda(), tabs.cuda(), targets.cuda().long()
                optimizer.zero_grad()

                tabs = None if nbr_tabular == 0 else tabs
                outputs = net(inputs, tabs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data.item()
                if use_cuda:
                    outputs, targets = outputs.cpu(), targets.cpu()

                preds.append(outputs.detach().numpy())
                ytrue.append(targets.detach().numpy())

                if adaptative_lr is not None:
                    scheduler.step()

            loss_train = running_loss/(batch_idx+1)
            acc_score_train = accuracy_score(np.vstack(preds).argmax(axis=1),
                                             np.hstack(ytrue))
            f1_score_train = f1_score(np.vstack(preds).argmax(axis=1),
                                      np.hstack(ytrue), average='weighted')
            # print('**********************************')
            print('train', epoch, acc_score_train, f1_score_train, loss_train)
            writer.add_scalar('Loss/train', loss_train, epoch)
            writer.add_scalar('Accuracy/train', acc_score_train, epoch)
            writer.add_scalar('F1/train', f1_score_train, epoch)

            # VALIDATION
            net.eval()
            test_loss = 0
            running_loss = 0.0
            preds = []
            ytrue = []
            for batch_idx, (inputs, tabs, targets) in enumerate(valloader):
                if use_cuda:
                    inputs, tabs, targets = inputs.cuda(), tabs.cuda(), targets.cuda().long()
                with torch.no_grad():
                    tabs = None if nbr_tabular == 0 else tabs
                    outputs = net(inputs, tabs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.data.item()

                    if use_cuda:
                        outputs, targets = outputs.cpu(), targets.cpu()

                    preds.append(outputs.detach().numpy())
                    ytrue.append(targets.detach().numpy())

                running_loss += loss.data.item()

            # print('val',running_loss, batch_idx)
            loss_val = running_loss/(batch_idx+1)
            acc_score_val = accuracy_score(np.vstack(preds).argmax(axis=1),
                                           np.hstack(ytrue))
            f1_score_val = f1_score(np.vstack(preds).argmax(axis=1),
                                    np.hstack(ytrue), average='weighted')
            # print('*********/////////')
            # print(np.vstack(preds).argmax(axis=1))
            # print(np.hstack(ytrue))

            print()

            # print('val',epoch,acc_score_val)
            writer.add_scalar('Loss/val', loss_val, epoch)
            writer.add_scalar('Accuracy/val', acc_score_val, epoch)
            writer.add_scalar('F1/val', f1_score_val, epoch)

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=loss_val, accuracy=acc_score_val,
                        f1_score=f1_score_val)


def test_classification(result, in_folder, in_labels, filenames_to_include=None,
                        filenames_to_exclude=None):
    if filenames_to_exclude is None:
        filenames_to_exclude = ['tot_commit2_weights.npy', 'sc_edge_normalized.npy',
                                'sc_vol_normalized.npy']
    loaded_stuff = load_data(directory_path=in_folder,
                             labels_path=in_labels,
                             features_filename_include=filenames_to_include,
                             features_filename_exclude=filenames_to_exclude)
    # Handle separately the test set
    transform = transforms.Compose([add_connections(), remove_connections()])
    testset_ori = ConnectomeDataset(loaded_stuff, mode='test',
                                    transform=False)
    testset_add_rem = ConnectomeDataset(loaded_stuff, mode='test',
                                        transform=transform)
    testset_noise = ConnectomeDataset(loaded_stuff, mode='test',
                                      transform=add_noise())
    testset_row_col = ConnectomeDataset(loaded_stuff, mode='test',
                                        transform=remove_row_column())
    testset = ConcatDataset([testset_ori])  # , testset_add_rem,
    #  testset_noise, testset_row_col])

    rng_cpu = torch.Generator()
    rng_cpu.manual_seed(2277)
    test_idx = list(range(len(testset)))
    class_w = balance_sampler(testset, test_idx)
    test_sampler = WeightedRandomSampler(test_idx, class_w,
                                         generator=rng_cpu)
    # test_sampler = SubsetRandomSampler(test_idx)
    set_seed()
    testloader = DataLoader(testset, batch_size=50,
                            num_workers=1, sampler=test_sampler,
                            worker_init_fn=seed_worker)

    best_trial = result.get_best_trial("loss", "min", "last")
    # Number of features / matrix size
    nbr_features = len(loaded_stuff[-1])
    matrix_size = loaded_stuff[-2]
    nbr_class = len(np.unique(loaded_stuff[1]))
    nbr_tabular = len(loaded_stuff[2][0]) if not np.any(
        np.isnan(loaded_stuff[2])) else 0
    net = BrainNetCNN_double(nbr_features, matrix_size, nbr_class, nbr_tabular,
                             l1=best_trial.config['l1'],
                             l2=best_trial.config['l2'],
                             l3=best_trial.config['l3'])
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=best_trial.config['lr'])

    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])
        # cudnn.benchmark = True

    best_checkpoint_dir = best_trial.checkpoint.value
    # print('best_checkpoint_dir', best_checkpoint_dir)
    model_state, _ = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    net.load_state_dict(model_state)
    # optimizer.load_state_dict(optimizer_state)

    test_loss = 0
    running_loss = 0.0

    preds = []
    ytrue = []

    for batch_idx, (inputs, tabs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, tabs, targets = inputs.cuda(), tabs.cuda(), targets.cuda().long()
        with torch.no_grad():
            tabs = None if nbr_tabular == 0 else tabs
            outputs = net(inputs, tabs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()

            if use_cuda:
                outputs, targets = outputs.cpu(), targets.cpu()

            preds.append(outputs.detach().numpy())
            ytrue.append(targets.detach().numpy())

        running_loss += loss.data.item()

    loss_test = running_loss/(batch_idx+1)
    acc_score_test = accuracy_score(np.vstack(preds).argmax(axis=1),
                                    np.hstack(ytrue))
    f1_score_test = f1_score(np.vstack(preds).argmax(axis=1),
                             np.hstack(ytrue), average='weighted')
    color_print(np.vstack(preds).argmax(axis=1))
    color_print(np.hstack(ytrue))
    print('Loss/test', loss_test)
    print('Accuracy/test', acc_score_test)
    print('F1/test', f1_score_test)
    print(best_trial.config)
