#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_from_tsfile_to_dataframe, load_from_tsfile
from sktime.datatypes._panel._convert import from_nested_to_2d_array


def load_dataset(dataset_name, dataset_type='UCR'):
    if dataset_type == 'UCR':
        train_x, train_y = load_from_tsfile_to_dataframe(
            './datasets/UCR/{dataset}/{dataset}_TRAIN.ts'.format(dataset=dataset_name))
        test_x, test_y = load_from_tsfile_to_dataframe(
            './datasets/UCR/{dataset}/{dataset}_TEST.ts'.format(dataset=dataset_name))

        le = LabelEncoder()
        le.fit(np.append(train_y, test_y))

        train_x, train_y = from_nested_to_2d_array(
            train_x).values, le.transform(train_y)
        test_x, test_y = from_nested_to_2d_array(
            test_x).values, le.transform(test_y)

        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
    else:
        train_x, train_y = load_from_tsfile(
            './datasets/UEA/{dataset}/{dataset}_TRAIN.ts'.format(
                dataset=dataset_name),
            return_data_type='numpy3d')
        test_x, test_y = load_from_tsfile(
            './datasets/UEA/{dataset}/{dataset}_TEST.ts'.format(dataset=dataset_name), return_data_type='numpy3d')

        le = LabelEncoder()
        le.fit(np.append(train_y, test_y))

        train_x, train_y = np.transpose(
            train_x, (0, 2, 1)), le.transform(train_y)
        test_x, test_y = np.transpose(test_x, (0, 2, 1)), le.transform(test_y)

    return train_x, train_y, test_x, test_y


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
