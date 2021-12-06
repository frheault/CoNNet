
#!/usr/bin/env python

import logging
import os
from CoNNet.utils import (color_print, load_data, ConnectomeDataset,
                          add_noise, add_connections, remove_connections,
                          remove_row_column, add_spike)
from CoNNet.models import BrainNetCNN
import torch
import numpy as np

import torch.nn.functional as F
import torch.nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import random_split

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

from ray import tune
import random

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
                         checkpoint_dir=None):
    loaded_stuff = load_data(directory_path=in_folder,
                             labels_path=in_labels,
                             features_filename_exclude=['tot_commit2_weights.npy',
                                                        'sc_edge_normalized.npy',
                                                        'sc_vol_normalized.npy'])

    # Prepare train/val with data augmentation
    # 1) Add and remove connections
    # 2) Add noise (+/-) to all existing connections
    transform = transforms.Compose([add_connections(), remove_connections()])
    trainset_ori = ConnectomeDataset(loaded_stuff, mode='train',
                                     transform=False)
    trainset_add_rem = ConnectomeDataset(loaded_stuff, mode='train',
                                         transform=transform)
    trainset_noise = ConnectomeDataset(loaded_stuff, mode='train',
                                       transform=add_noise())
    trainset_spike = ConnectomeDataset(loaded_stuff, mode='train',
                                       transform=add_spike())
    trainset_row_col = ConnectomeDataset(loaded_stuff, mode='train',
                                         transform=remove_row_column())
    trainset = ConcatDataset([trainset_ori, trainset_add_rem,
                              trainset_noise, trainset_spike,
                              trainset_row_col])

    # Split training set in two (train/validation)
    len_ts = len(trainset)
    rng = torch.Generator().manual_seed(42)
    test_abs = int(len_ts * 0.80)
    train_subset, val_subset = random_split(trainset,
                                            [test_abs, len_ts - test_abs],
                                            generator=rng)
    color_print('Final datasets (with data augmentation): {} train and {} val'.format(
        len(train_subset), len(val_subset)))
    set_seed()
    trainloader = DataLoader(train_subset, batch_size=int(config['batch_size']),
                             shuffle=True, num_workers=1,
                             worker_init_fn=seed_worker)

    valloader = DataLoader(val_subset, batch_size=int(config['batch_size']),
                           shuffle=True, num_workers=1,
                           worker_init_fn=seed_worker)
    # Number of features / matrix size
    nbr_features = loaded_stuff[1].shape[1]
    matrix_size = loaded_stuff[1].shape[2]
    nbr_class = np.max(len(np.unique(loaded_stuff[0])))
    nbr_tabular = len(loaded_stuff[2][0]) if not np.any(
        np.isnan(loaded_stuff[2])) else 0

    net = BrainNetCNN(nbr_features, matrix_size, nbr_class, nbr_tabular,
                      l1=config['l1'], l2=config['l2'], l3=config['l3'])

    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])
        # cudnn.benchmark = True

    momentum = 0.9
    lr = config['lr']
    wd = 0.0005

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                momentum=momentum, weight_decay=wd)

    if checkpoint_dir:
        filename = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.isfile(filename):
            model_state, optimizer_state = torch.load(filename)
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    writer = SummaryWriter()
    for epoch in range(num_epoch):
        net.train()
        running_loss = 0.0
        preds = []
        ytrue = []
        for batch_idx, (inputs, tabs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, tabs, targets = inputs.cuda(), tabs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, tabs, targets = Variable(inputs), Variable(
                tabs).long(), Variable(targets).long()

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

        loss_train = running_loss/(batch_idx+1)
        acc_score_train = accuracy_score(np.vstack(preds).argmax(axis=1),
                                         np.hstack(ytrue))
        # print('train',epoch,acc_score_train)
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Accuracy/train', acc_score_train, epoch)

        # VALIDATION
        net.eval()
        test_loss = 0
        running_loss = 0.0
        preds = []
        ytrue = []
        for batch_idx, (inputs, tabs, targets) in enumerate(valloader):
            if use_cuda:
                inputs, tabs, targets = inputs.cuda(), tabs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, tabs, targets = Variable(inputs), Variable(
                    tabs), Variable(targets).long()

                tabs = None if nbr_tabular == 0 else tabs
                outputs = net(inputs, tabs)
                loss = criterion(outputs, targets)

                test_loss += loss.data.item()

                if use_cuda:
                    outputs, targets = outputs.cpu(), targets.cpu()

                preds.append(outputs.numpy())
                ytrue.append(targets.numpy())

            running_loss += loss.data.item()

        # print('val',running_loss, batch_idx)
        loss_val = running_loss/(batch_idx+1)
        acc_score_val = accuracy_score(np.vstack(preds).argmax(axis=1),
                                       np.hstack(ytrue))
        # print('val',epoch,acc_score_val)
        writer.add_scalar('Loss/val', loss_val, epoch)
        writer.add_scalar('Accuracy/val', acc_score_val, epoch)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss_val, accuracy=acc_score_val)


def test_classification(result, in_folder, in_labels):
    loaded_stuff = load_data(directory_path=in_folder,
                             labels_path=in_labels,
                             features_filename_exclude=['tot_commit2_weights.npy',
                                                        'sc_edge_normalized.npy',
                                                        'sc_vol_normalized.npy'])
    # Handle separately the test set
    testset = ConnectomeDataset(loaded_stuff, mode='test',
                                transform=False)
    set_seed()
    testloader = DataLoader(testset, batch_size=10,
                            shuffle=True, num_workers=1,
                            worker_init_fn=seed_worker)

    best_trial = result.get_best_trial("loss", "min", "last")
    # Number of features / matrix size
    nbr_features = loaded_stuff[1].shape[1]
    matrix_size = loaded_stuff[1].shape[2]
    nbr_class = np.max(len(np.unique(loaded_stuff[0])))
    nbr_tabular = len(loaded_stuff[2][0]) if not np.any(
        np.isnan(loaded_stuff[2])) else 0
    net = BrainNetCNN(nbr_features, matrix_size, nbr_class, nbr_tabular,
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
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    net.load_state_dict(model_state)
    # optimizer.load_state_dict(optimizer_state)

    test_loss = 0
    running_loss = 0.0

    preds = []
    ytrue = []

    for batch_idx, (inputs, tabs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, tabs, targets = inputs.cuda(), tabs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, tabs, targets = Variable(inputs), Variable(
                tabs), Variable(targets).long()

            tabs = None if nbr_tabular == 0 else tabs
            outputs = net(inputs, tabs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()

            if use_cuda:
                outputs, targets = outputs.cpu(), targets.cpu()

            preds.append(outputs.numpy())
            ytrue.append(targets.numpy())

        running_loss += loss.data.item()

    loss_test = running_loss/batch_idx
    acc_score_test = accuracy_score(np.vstack(preds).argmax(axis=1),
                                    np.hstack(ytrue))

    print('Loss/test', loss_test)
    print('Accuracy/test', acc_score_test)
    print(best_trial.config)
