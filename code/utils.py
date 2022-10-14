#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def load_dataset(dataset_name):
    train_x, train_y = load_from_tsfile_to_dataframe(
        './datasets/Univariate2018_ts/{dataset}/{dataset}_TRAIN.ts'.format(dataset=dataset_name))
    test_x, test_y = load_from_tsfile_to_dataframe(
        './datasets/Univariate2018_ts/{dataset}/{dataset}_TEST.ts'.format(dataset=dataset_name))
    le = LabelEncoder()
    le.fit(np.append(train_y, test_y))

    train_x, train_y = from_nested_to_2d_array(train_x).values, le.transform(train_y)
    test_x, test_y = from_nested_to_2d_array(test_x).values, le.transform(test_y)

    train_x = TimeSeriesScalerMeanVariance().fit_transform(train_x)
    test_x = TimeSeriesScalerMeanVariance().fit_transform(test_x)

    return train_x, train_y, test_x, test_y
