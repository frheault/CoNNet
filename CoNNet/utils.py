#!/usr/bin/env python

from sklearn.model_selection import train_test_split
import torch.utils.data.dataset
from sklearn.preprocessing import normalize
import os
import pandas as pd
import numpy as np
import random

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
              adjacency_filename='len.npy',
              features_filename_include=None,
              features_filename_exclude=None):
    adjacency_matrices = []
    features_matrices = []
    labels_data = pd.read_excel(labels_path, index_col=0)
    subj_id = labels_data.index.tolist()
    labels = labels_data['labels'].tolist()

    # Shuffle the ordering
    tmp = list(zip(subj_id, labels))
    random.shuffle(tmp)
    subj_id, labels = zip(*tmp)

    if features_filename_include is None:
        features_filename_include = []
        for filename in os.listdir(os.path.join(directory_path, subj_id[0])):
            _, ext = os.path.splitext(filename)
            if ext in ['.npy', '.txt']:#and filename != adjacency_filename:
                features_filename_include.append(filename)
    if features_filename_exclude is not None:
        for filename in features_filename_exclude:
            features_filename_include.remove(filename)

    for subj in subj_id:
        adjacency_matrix = read_matrix(os.path.join(directory_path, subj,
                                                    adjacency_filename))
        features = np.zeros(adjacency_matrix.shape +
                            (len(features_filename_include),))
        for i, filename in enumerate(features_filename_include):
            
            features[:, :, i] = read_matrix(os.path.join(directory_path, subj,
                                                         filename))

        adjacency_matrices.append(adjacency_matrix)
        features_matrices.append(features)

    labels = np.array(labels, dtype=np.float)
    adjacency_matrices = np.array(adjacency_matrices, dtype=np.float)
    features_matrices = np.array(features_matrices, dtype=np.float)
    features_matrices = np.swapaxes(features_matrices, 1,3)

    return subj_id, labels, adjacency_matrices, features_matrices


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
        _, labels, adjacency_matrices, features_matrices = loaded_data
        self.mode = mode
        self.transform = transform


        idx_train = range(120)
        idx_val = range(120, 200)
        idx_train_val = range(200)
        idx_test = range(200, 264)

        if self.mode == "train":
            x = features_matrices[idx_train, ...]
            y = labels[idx_train, ...]
            # adj = adjacency_matrices[idx_train, ...]
        elif self.mode == "validation":
            x = features_matrices[idx_val, ...]
            y = labels[idx_val, ...]
            # adj = adjacency_matrices[idx_val, ...]
        else:
            x = features_matrices[idx_train_val, ...]
            y = labels[idx_train_val, ...]
            # adj = adjacency_matrices[idx_train_val, ...]

        self.X = torch.FloatTensor(x.astype(np.float32))
        self.Y = torch.FloatTensor(y.astype(np.float32))
        # self.ADJ = torch.FloatTensor(adj.astype(np.float32))

        print(self.mode, self.X.shape, (self.Y.shape))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample
