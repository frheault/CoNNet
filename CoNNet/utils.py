#!/usr/bin/env python

from copy import deepcopy
import itertools
from sklearn.model_selection import train_test_split
import torch.utils.data.dataset
from sklearn.preprocessing import normalize
import os
import pandas as pd
import numpy as np
import random
import logging
import ray
import coloredlogs
from termcolor import colored

verbose = True


def color_print(txt, color='green'):
    if verbose:
        print(colored(txt, color))


def read_matrix(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)

    if 'vol' in filepath or 'sc' in filepath:
        data[data > 0] = np.log10(data[data > 0])

    return data / np.max(data)


def load_data(directory_path, labels_path,
              features_filename_include=None,
              features_filename_exclude=None,
              as_one_hot=False):
    features_matrices = []
    labels_data = pd.read_excel(labels_path, index_col=0)
    drop_subj = 0
    for subj in labels_data.index.tolist():
        if not os.path.isdir(os.path.join(directory_path, subj)):
            labels_data = labels_data.drop([subj])
            drop_subj += 1

    if drop_subj:
        color_print(
            '{} drop subjects from missing files/folders.'.format(drop_subj))

    subj_id = labels_data.index.tolist()
    labels = labels_data['labels'].tolist()

    if len(labels_data.columns) > 1:
        nbr_tab = 0
        extra_tabular = []
        for i in range(1, len(labels_data.columns)):
            tab = labels_data[labels_data.columns[i]].tolist()
            extra_tabular.append(tab)
            nbr_tab += 1
        color_print('Found {} tabular values.'.format(nbr_tab))
    else:
        extra_tabular = [[None] for i in range(len(labels))]
        nbr_tab = 1
    extra_tabular = np.array(extra_tabular).reshape(
        (len(labels), nbr_tab)).tolist()

    # Shuffle the ordering
    tmp = list(zip(subj_id, labels, extra_tabular))
    random.seed(0)
    random.shuffle(tmp)
    subj_id, labels, extra_tabular = zip(*tmp)

    if features_filename_include is None:
        features_filename_include = []
        for filename in os.listdir(os.path.join(directory_path, subj_id[0])):
            _, ext = os.path.splitext(filename)
            if ext in ['.npy', '.txt']:
                features_filename_include.append(filename)
    if features_filename_exclude is not None:
        for filename in features_filename_exclude:
            if filename in features_filename_include:
                features_filename_include.remove(filename)

    for subj in subj_id:
        base_matrix = read_matrix(os.path.join(directory_path, subj,
                                               features_filename_include[0]))
        features = np.zeros(base_matrix.shape +
                            (len(features_filename_include),))
        for i, filename in enumerate(features_filename_include):
            features[:, :, i] = read_matrix(os.path.join(directory_path, subj,
                                                         filename))

        features_matrices.append(features)

    labels = np.array(labels, dtype=np.int64)
    if as_one_hot:
        tmp_labels = np.zeros((len(labels), np.max(labels)))
        for i in range(len(labels)):
            tmp_labels[i, labels[i]-1] = 1
        labels = tmp_labels

    features_matrices = np.array(features_matrices, dtype=np.float)
    features_matrices = np.swapaxes(features_matrices, 1, 3)
    extra_tabular = np.array(extra_tabular, dtype=np.float)

    return labels, features_matrices, extra_tabular


class ConnectomeDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data,
                 mode="train",
                 transform=False,
                 class_balancing=False):
        """
        Args:
            directory (string): Path to the dataset.
            mode (str): train = 90% Train, validation=10% Train, train+validation=100% train else test.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        labels, features_matrices, extra_tabular = loaded_data
        self.mode = mode
        self.transform = transform

        split_ratio = 0.75
        idx_train = list(range(int(len(labels)*split_ratio)))
        idx_test = list(range(int(len(labels)*split_ratio), len(labels)))
        func_name = 'a' if transform else 'no'
        color_print('Creating loader with {} datasets for {} set with {} transform'.format(
            len(labels), mode, func_name))

        if self.mode == 'train':
            x = features_matrices[idx_train, ...]
            y = labels[idx_train, ...]
            t = extra_tabular[idx_train, ...]
        elif self.mode == 'test':
            x = features_matrices[idx_test, ...]
            y = labels[idx_test, ...]
            t = extra_tabular[idx_test, ...]

        self.X = torch.FloatTensor(x.astype(np.float32))
        self.Y = torch.FloatTensor(y.astype(np.float32))
        self.T = torch.FloatTensor(t.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def transform():
        return

    def __getitem__(self, idx):
        sample = [self.X[idx], self.T[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample


class add_noise(object):
    """ Add noise to existing connections, centered at 0 with STD of 0.025 """

    def __call__(self, array):
        # np.random.seed(0)
        for i in range(len(array)):
            tmp_arr = array[i].numpy()
            shape = tmp_arr.shape
            noise = np.random.normal(0, 0.025,
                                     np.prod(shape)).reshape(shape)
            minval = np.min(tmp_arr[np.nonzero(tmp_arr)])
            noise[tmp_arr < minval] = 0
            noise = np.triu(noise) + np.triu(noise, k=1).T
            tmp_arr += noise

        return array


class remove_row_column(object):
    """ Add noise to existing connections, centered at 0 with STD of 0.025 """

    def __call__(self, array):
        # np.random.seed(0)
        idx = int(np.random.rand() * array[0].numpy().shape[0])
        for i in range(len(array)):
            tmp_arr = array[i].numpy()
            tmp_arr[:, idx] = 0
            tmp_arr[idx, :] = 0

        return array


class add_connections(object):
    """ Add connections to matrices, +1% new connections with positive
        values similar to the noise (above) """

    def __call__(self, array):
        # np.random.seed(0)
        tmp_arr = array.numpy()
        
        shape = tmp_arr.shape[1:3]
        total_new_conn = np.prod(shape) // 100
        positions = random.sample(np.argwhere(tmp_arr[0] == 0).tolist(),
                                  total_new_conn)
        for pos in positions:
            noise = np.abs(np.random.normal(0, 0.05, len(tmp_arr)))
            tmp_arr[:, pos[0], pos[1]] = noise
            tmp_arr[:, pos[1], pos[0]] = noise

        return array


class remove_connections(object):
    """ Removes connections to matrices, -1% new connections force to zero """

    def __call__(self, array):
        # np.random.seed(0)
        tmp_arr = array.numpy()
        shape = tmp_arr.shape[1:3]
        total_new_conn = np.prod(shape) // 100
        positions = random.sample(np.argwhere(tmp_arr[0] > 0).tolist(),
                                  total_new_conn)
        for pos in positions:
            tmp_arr[:, pos[0], pos[1]] = 0
            tmp_arr[:, pos[1], pos[0]] = 0

        return array


class add_spike(object):
    """ Add connections to matrices, +1% new connections with positive
        values similar to the noise (above) """

    def __call__(self, array):
        # np.random.seed(0)
        tmp_arr = array.numpy()
        pos = int(np.random.rand()*tmp_arr.shape[1]), int(np.random.rand()*tmp_arr.shape[1])
        which_array = int(np.random.rand() * len(tmp_arr))

        tmp_arr[which_array, pos[0], pos[1]] = 10
        tmp_arr[which_array, pos[1], pos[0]] = 10
        tmp_arr[which_array, :, :] /= 10

        return array