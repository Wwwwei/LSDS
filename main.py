#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import math
import numpy as np
import torch

from code.utils import load_dataset
from code.model import LSDSModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # =============args=============
    dataset_name = 'Coffee'
    epochs = 200
    L = 0.1
    R = 3
    H = 64
    dropout = 0.
    batch_size = 16
    lr = 0.001
    num_workers = 0
    # ==============================

    train_x, train_y, test_x, test_y = load_dataset(
        dataset_name=dataset_name, dataset_type='UCR')

    num_classes = max(set(test_y.squeeze().tolist() +
                      train_y.squeeze().tolist())) + 1
    num_timesteps = train_x.shape[1]
    num_channels = train_x.shape[-1]

    print('dataset:', dataset_name)
    print('GPU is available:', torch.cuda.is_available())

    print('num_classes:', num_classes)
    print('num_timesteps:', num_timesteps)
    print('num_channels:', num_channels)

    print('train_x:', train_x.shape, 'train_y:', train_y.shape)
    print('test_x:', test_x.shape, 'test_y:', test_y.shape)

    print('shapelets_len:', int(num_timesteps * L))
    print('shapelets_stride:', R)
    print('shapelets_num:',
          math.floor((num_timesteps - int(num_timesteps * L)) / R + 1))

    model = LSDSModel(
        L=L,
        R=R,
        H=H,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        num_channels=num_channels,
        num_workers=num_workers,
        num_classes=num_classes
    )
    model.train(train_x, train_y)
    model.predict(test_x, test_y)
