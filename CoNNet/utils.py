#!/usr/bin/env python

import os
import random
from termcolor import colored

import numpy as np
import pandas as pd
import torch.utils.data.dataset

verbose = True


def color_print(txt, color='green'):
    if verbose:
        print(colored(txt, color))


def read_matrix(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath).astype(np.float64)
    elif ext == '.npy':
        data = np.load(filepath).astype(np.float64)

    # testing
    # mask = np.load('/home/frheault/Datasets/learning_ml/taylor/CAARE_reorganized/mask.npy')
    # mask = np.load('/home/frheault/Datasets/learning_ml/barry_connectome/BLSA_CAM_CAM/mask.npy')
    # data *= mask

    return data / np.percentile(data[data > 0.00001], 50)


def load_data(directory_path, labels_path,
              features_filename_include=None,
              features_filename_exclude=None,
              as_one_hot=False, how_many=100):
    labels_data = pd.read_excel(labels_path, index_col=0)
    drop_subj = 0
    labels_list = labels_data.index.tolist()
    random.shuffle(labels_list)
    for i, subj in enumerate(labels_list):
        if not os.path.isdir(os.path.join(directory_path, subj)) or i > how_many:
            labels_data = labels_data.drop([subj])
            drop_subj += 1

    if drop_subj:
        color_print(
            '{} drop subjects from missing files/folders out of {}.'.format(
                drop_subj, len(labels_list)))

    subj_id = labels_data.index.tolist()
    labels = labels_data['labels'].tolist()
    pairing = labels_data['pairing'].tolist()

    if len(labels_data.columns) > 1:
        nbr_tab = 0
        extra_tabular = []
        for i in range(2, len(labels_data.columns)):
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
    tmp = list(zip(subj_id, labels, extra_tabular, pairing))
    random.seed(0)
    random.shuffle(tmp)
    subj_id, labels, extra_tabular, pairing = zip(*tmp)

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

    color_print('Loading data using {} features'.format(
        len(features_filename_include)))
    color_print(features_filename_include)

    matrix_size = read_matrix(os.path.join(directory_path, subj_id[0],
                                           features_filename_include[0])).shape[0]

    labels = np.array(labels, dtype=np.int64)
    extra_tabular = np.array(extra_tabular, dtype=np.float64)

    subj_list = [os.path.join(directory_path, subj) for subj in subj_id]
    return subj_list, labels, extra_tabular, pairing, matrix_size, features_filename_include


def balance_sampler(dataset, idx):
    labels = []
    for i in idx:
        labels.append(int(dataset[i][2]))

    labels = np.array(labels)
    uniq = np.unique(labels)
    w = np.zeros(len(uniq))
    prob_dist = np.zeros(len(labels), dtype=float)

    for i, u in enumerate(uniq):
        w[i] = 1 - len(np.where(labels == u)[0]) / len(labels)
        prob_dist[labels == u] = w[i]
        # print(i, u, len(np.where(labels == u)[0]), len(labels))

    print()
    color_print('Rebalanced sampler with probability {} for class {}'.format(
        w, uniq))
    print()
    return prob_dist


class ConnectomeDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data,
                 mode="train",
                 transform=False,
                 class_balancing=False):
        """
        Args:

        """
        subj_list, labels, extra_tabular, pairing, \
            self.matrix_size, self.features_filename = loaded_data
        self.mode = mode
        self.transform = transform

        split_ratio = 0.75
        # Since pair of session are allowed, both sessions must be in the same
        # set of data. So we select IDs based on the unique individual, not the
        # session
        idx_train = list(range(int(len(np.unique(pairing))*split_ratio)))
        idx_test = list(range(int(len(np.unique(pairing))*split_ratio),
                              len(np.unique(pairing))))
        func_name = 'a' if transform else 'no'
        if self.mode == 'train':
            true_idx = []
            for idx in idx_train:
                tmp = np.argwhere(np.array(pairing) == pairing[idx]).ravel()
                true_idx.extend(tmp)
                # true_idx.append(int(tmp[0]))
        elif self.mode == 'test':
            true_idx = []
            for idx in idx_test:
                tmp = np.argwhere(np.array(pairing) == pairing[idx]).ravel()
                true_idx.extend(tmp)
                # true_idx.append(int(tmp[0]))

        # If session stuff?
        # true_idx = list(set(true_idx))

        x = [subj_list[i] for i in true_idx]
        y = labels[true_idx, ...]
        t = extra_tabular[true_idx, ...]

        color_print('Creating loader with {} datasets for {} set with {} '
                    'transform'.format(len(y), mode, func_name))

        self.X = x
        self.Y = torch.FloatTensor(y.astype(np.int64))
        self.T = torch.FloatTensor(t.astype(np.float64))


    def __len__(self):
        return len(self.X)

    def transform():
        return

    def __getitem__(self, idx):
        path = self.X[idx]
        features = np.zeros((len(self.features_filename),
                             self.matrix_size, self.matrix_size),
                            dtype=np.float64)
        for i, filename in enumerate(self.features_filename):
            features[i, :, :] = read_matrix(os.path.join(path,
                                                         filename))
        features = torch.FloatTensor(features)
        # extra_tabular = np.array(self.T, dtype=np.float)
        if self.transform:
            features = self.transform(features)

        sample = [features, self.T[idx], self.Y[idx]]
        return sample


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class add_noise(object):
    """ Add noise to existing connections, centered at 0 with STD of 0.025 """

    def __call__(self, array):
        # np.random.seed(0)
        for i in range(len(array)):
            tmp_arr = array[i].numpy()
            noise_level = np.std(tmp_arr[tmp_arr > 0])
            shape = tmp_arr.shape
            noise = np.random.normal(0, noise_level,
                                     np.prod(shape)).reshape(shape)
            minval = np.min(tmp_arr[np.nonzero(tmp_arr)])
            noise[tmp_arr < minval] = 0
            noise = np.triu(noise) + np.triu(noise, k=1).T
            tmp_arr += noise

        return array


class remove_row_column(object):
    """ """

    def __call__(self, array):
        # np.random.seed(0)
        num_row_col = array[0].numpy().shape[0]
        to_remove = max(int(num_row_col / 50), 1)
        idx_list = np.random.rand(to_remove) * num_row_col
        for idx in idx_list:
            idx = int(idx)
            tmp_arr = array.numpy()
            tmp_arr[:, :, idx] = 0
            tmp_arr[:, idx, :] = 0

        return array


class add_connections(object):
    """ Add connections to matrices, +1% new connections with positive
        values similar to the noise (above) """

    def __call__(self, array):
        # np.random.seed(0)
        tmp_arr = array.numpy()

        positions = np.argwhere(tmp_arr[0] == 0).tolist()
        total_new_conn = len(positions) // 10

        positions = random.sample(positions,
                                  min(len(positions), total_new_conn))
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
        positions = np.argwhere(tmp_arr[0] > 0).tolist()
        total_new_conn = len(positions) // 10 if len(positions) else 0
        positions = random.sample(positions,
                                  min(len(positions), total_new_conn))
        for pos in positions:
            tmp_arr[:, pos[0], pos[1]] = 0
            tmp_arr[:, pos[1], pos[0]] = 0

        return array
