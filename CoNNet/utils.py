#!/usr/bin/env python

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

verbose=True
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
    for subj in labels_data.index.tolist():
        if not os.path.isdir(os.path.join(directory_path, subj)):
            labels_data = labels_data.drop([subj])

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
    extra_tabular = np.array(extra_tabular).reshape((len(labels), nbr_tab)).tolist()

    # Shuffle the ordering
    tmp = list(zip(subj_id, labels, extra_tabular))
    random.seed(0)
    random.shuffle(tmp)
    subj_id, labels, extra_tabular = zip(*tmp)
    # subj_id, labels = list(subj_id), list(labels)

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
        #if os.path.isdir(os.path.join(directory_path, subj)):
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

        # ADAPT
        split_ratio = 0.75
        idx_train = list(range(int(len(labels)*split_ratio)))
        idx_test = list(range(int(len(labels)*split_ratio), len(labels)))
        color_print('Load {} datasets, split with ratio of {}%.'.format(
            len(labels), split_ratio*100))
        color_print('{} datasets in train and {} in test.'.format(
            len(idx_train), len(idx_test)))

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
        np.random.seed(0)
        for i in range(len(array)):
            tmp_arr = array[i].numpy()
            non_zeros = np.count_nonzero(tmp_arr)
            tmp_arr[tmp_arr > 0] += np.random.normal(0, 0.025, non_zeros)

        return array


class add_connections(object):
    """ Add connections to matrices, +1% new connections with positive
        values similar to the noise (above) """

    def __call__(self, array):
        np.random.seed(0)
        tmp_arr = array.numpy()
        shape = tmp_arr.shape[1:3]
        total_new_conn = np.prod(shape) // 100
        positions = random.sample(np.argwhere(tmp_arr[0] == 0).tolist(),
                                  total_new_conn)
        for pos in positions:
            tmp_arr[:, pos[0], pos[1]] = np.abs(np.random.normal(
                0, 0.05, len(tmp_arr)))

        return array


class remove_connections(object):
    """ Removes connections to matrices, -1% new connections force to zero """

    def __call__(self, array):
        np.random.seed(0)
        tmp_arr = array.numpy()
        shape = tmp_arr.shape[1:3]
        total_new_conn = np.prod(shape) // 100
        positions = random.sample(np.argwhere(tmp_arr[0] > 0).tolist(),
                                  total_new_conn)
        for pos in positions:
            tmp_arr[:, pos[0], pos[1]] = 0

        return array
