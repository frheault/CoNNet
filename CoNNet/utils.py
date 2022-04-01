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


def read_matrix(filepath, mask_path=None):
    dirname = '/home/frheault/Datasets/learning_ml/pre_training_connectome/all_data/'
    filename = 'mask.npy'
    mask_path = os.path.join(dirname, filename)
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath).astype(np.float64)
    elif ext == '.npy':
        data = np.load(filepath).astype(np.float64)

    if mask_path is not None:
        mask = np.load(mask_path)
        data *= mask

    return data / np.percentile(data[data > 0.00001], 50)


def load_data(directory_path, labels_path,
              features_filename_include=None,
              features_filename_exclude=None,
              how_many=100):
    # TODO Add an example spreadsheet and data (Like the scanner prediction)
    # TODO retry filename exclude
    """
    Prepare data to be loaded by the Pytorch Dataset class (ConnectomeDataset).
    Parse a spreadsheet and prepare list of subject name, pairing, tasks
    (classes, values and tabular) and path to matrices for future loading.

    Parameters
    ----------
    directory_path : str
       Absolute path to the folder containing all subjects and their
       connectivity matrices. Connectivity matrices should have an identical
       naming convention across subject (i.e no ${SUBJ_ID}__fa.npy).
    labels_path : str
        Path to excel spreadsheet containing all subject ID (matching folder
        in directory_path) and their relative information.

        Column names should be:
        subjects, pairing, classification_NAME, regression_NAME, tabular_NAME
        Pairing identifies same-subjects (starting from 1-N.
        The network can be scale to perform multiple classification and
        regression tasks. But the have to be identified as classification_XXXX
        or regression_YYYY or tabular_ZZZZ with unique name for XXXX/YYYY/ZZZZ
        with a single underscore (between the kind of task and its unique name)

    features_filename_include : list (str)
        Include only the matrices matching these filenames and exclude all
        others.
    features_filename_exclude : list (str)
        Exclude only the matrices matching these filenames and include all
        others.
        Length of sum(nbr_classes_each) + nbr_regression
    how_many : int
        Limits the amount of subject to load, following the order of the
        provided spreadsheet. Be careful if ordered in any meaningful way.

    Returns
    -------
    tuple (7,)
        subj_list: list (str)
        pairing: list (int)
        extra_classification: np.ndarray (nbr_subj, nbr_classification)
        extra_regression: np.ndarray (nbr_subj, nbr_regression)
        extra_tabular: np.ndarray (nbr_subj, nbr_tabular)
        matrix_size: int
        features_filename_include: list (str)

        All theses match the spreadsheet information, excluded missing data,
        is shuffled and ready for the class ConnectomeDataset.
    """
    labels_data = pd.read_excel(labels_path, index_col=0)
    drop_subj = 0
    labels_list = [str(i) for i in labels_data.index.tolist()]
    random.shuffle(labels_list)
    for i, subj in enumerate(labels_list):
        if not os.path.isdir(os.path.join(directory_path, subj)) \
                or i > how_many:
            labels_data = labels_data.drop([subj])
            drop_subj += 1

    if drop_subj:
        color_print(
            '{} drop subjects from missing files/folders out of {}.'.format(
                drop_subj, len(labels_list)))

    subj_id = labels_data.index.tolist()
    pairing = labels_data['pairing'].tolist()

    nbr_classifification, extra_classification = 0, []
    nbr_regression, extra_regression = 0, []
    nbr_tab, extra_tabular = 0, []
    for i in range(1, len(labels_data.columns)):
        full_name = labels_data.columns[i]
        task, name = full_name.split('_')
        if task == 'classification':
            tmp = labels_data[labels_data.columns[i]].tolist()
            nbr_classifification += 1
            extra_classification.append(tmp)
        elif task == 'regression':
            if name == 'age':
                tmp = (labels_data[labels_data.columns[i]] - 65.2622) / 9.4154
            tmp = tmp.tolist()
            nbr_regression += 1
            extra_regression.append(tmp)
        elif task == 'tabular':
            tmp = labels_data[labels_data.columns[i]].tolist()
            extra_tabular.append(tmp)
            nbr_tab += 1
        else:
            raise ValueError('Review naming convention of column.')

    color_print('Found {} classifcation values.'.format(nbr_classifification))
    color_print('Found {} regression values.'.format(nbr_regression))
    color_print('Found {} tabular values.'.format(nbr_tab))
    if nbr_classifification == 0:
        extra_classification = [[None] for i in range(len(subj_id))]
        nbr_classifification = 1
    if nbr_regression == 0:
        extra_regression = [[None] for i in range(len(subj_id))]
        nbr_regression = 1
    if nbr_tab == 0:
        extra_tabular = [[None] for i in range(len(subj_id))]
        nbr_tab = 1

    extra_classification = np.array(extra_classification).reshape(
        (len(subj_id), nbr_classifification)).tolist()
    extra_regression = np.array(extra_regression).reshape(
        (len(subj_id), nbr_regression)).tolist()
    extra_tabular = np.array(extra_tabular).reshape(
        (len(subj_id), nbr_tab)).tolist()

    # Shuffle the ordering
    tmp = list(zip(subj_id, pairing, extra_classification,
               extra_regression, extra_tabular))
    random.seed(0)
    random.shuffle(tmp)
    subj_id, pairing, extra_classification, extra_regression, \
        extra_tabular = zip(*tmp)

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

    matrix_size = read_matrix(os.path.join(
        directory_path, subj_id[0], features_filename_include[0])).shape[0]

    extra_classification = np.array(extra_classification, dtype=np.float64)
    extra_regression = np.array(extra_regression, dtype=np.float64)
    extra_tabular = np.array(extra_tabular, dtype=np.float64)

    subj_list = [os.path.join(directory_path, subj) for subj in subj_id]
    return subj_list, pairing, extra_classification, extra_regression, \
        extra_tabular, matrix_size, features_filename_include


def balance_sampler(dataset, idx):
    # TODO Balance only the first classification task?
    """
    If only one classification task is present, balance the classes so the
    sampler can pick both classes uniformly.

    Parameters
    ----------
    dataset : ConcatDataset
        Concatenated dataset (after data augmentation)
    idx : list (int)
        Indices that identifies the current set (train vs test)

    Returns
    -------
    prob_dist : np.ndarray
        Weights to balance the sampler. Rare classes has higher probability of
        being picked.
    """
    labels = []
    for i in idx:
        labels.append(float(dataset[i][2]))

    labels = np.array(labels)
    uniq = np.unique(labels)
    w = np.zeros(len(uniq))
    prob_dist = np.zeros(len(labels), dtype=float)

    for i, u in enumerate(uniq):
        w[i] = 1 - len(np.where(labels == u)[0]) / len(labels)
        prob_dist[labels == u] = w[i]

    print()
    color_print('Rebalanced sampler with probability {} for class {}'.format(
        w, uniq))
    print()
    return prob_dist


class ConnectomeDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data,
                 mode="train",
                 transform=False,
                 allow_duplicate_subj=True):
        """
        If only one classification task is present, balance the classes so the
        sampler can pick both classes uniformly.

        Parameters
        ----------
        loaded_data : tuple (7,)
            All information from the spreadsheet (see load_data() function)
        mode : str
            Either 'train' or 'test'. Split out the set early to avoid data
            contamination.
        transform: torchvision.transform
            Transformation function to apply for data_augmentation.
        allow_duplicate_subj: bool
            Using the pairing information, either use only the first of all
            duplicated subjects (False) or all of them (True).
        """

        subj_list, pairing, extra_classification, extra_regression, \
            extra_tabular, self.matrix_size, self.features_filename = loaded_data
        self.mode = mode
        self.transform = transform

        self.nbr_classification = len(extra_classification) if not np.any(
            np.isnan(extra_classification)) else 0
        self.nbr_regression = len(extra_regression) if not np.any(
            np.isnan(extra_regression)) else 0
        self.nbr_tabular = len(extra_tabular) if not np.any(
            np.isnan(extra_tabular)) else 0

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
                if allow_duplicate_subj:
                    true_idx.extend(tmp)
                else:
                    true_idx.append(int(tmp[0]))
        elif self.mode == 'test':
            true_idx = []
            for idx in idx_test:
                tmp = np.argwhere(np.array(pairing) == pairing[idx]).ravel()
                if allow_duplicate_subj:
                    true_idx.extend(tmp)
                else:
                    true_idx.append(int(tmp[0]))

        # If session stuff?
        true_idx = true_idx if allow_duplicate_subj else list(set(true_idx))

        filepaths = [subj_list[i] for i in true_idx]
        y_c = extra_classification[true_idx, :]
        y_r = extra_regression[true_idx, :]
        t = extra_tabular[true_idx, :]

        color_print('Creating loader with {} datasets for {} set with {} '
                    'transform'.format(len(filepaths), mode, func_name))

        self.filepaths = filepaths
        self.Y_c = torch.FloatTensor(y_c.astype(np.int64))
        self.Y_r = torch.FloatTensor(y_r.astype(np.float64))
        self.T = torch.FloatTensor(t.astype(np.float64))

    def __len__(self):
        return len(self.filepaths)

    def transform():
        return

    def __getitem__(self, idx):
        path = self.filepaths[idx]
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

        sample = [features, self.T[idx], self.Y_c[idx], self.Y_r[idx]]
        return sample


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class add_noise(object):
    """ Add noise to existing connections, centered at 0 with STD of relative
    to each matrice (estimate using the STD of values > 0) """

    def __call__(self, array):
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
    """ Randomly pick 2% of rows/columns and set them to zeros """

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
    """ Add connections to matrices, Count the number of missing connections
        and adds 10% of that value as new connections with positive
        values low noise """

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
    """ Remove connections to matrices, Count the number of existing connections
        and removes 10% of that value (set to 0) """

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
