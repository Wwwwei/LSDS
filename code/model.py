#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import torch
import torchmetrics
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import EarlyStopping





class TSCDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Encoder(nn.Module):
    def __init__(
            self,
            name='Encoder',
            input_dim=1,
            output_dim=32,
            num_layers=1,
            bidirectional=True,
            k=50
    ):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=output_dim // 2 if bidirectional else output_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.pool = nn.AdaptiveMaxPool1d(k)
        self.ln = nn.LayerNorm(output_dim)
        self.input_fc = nn.Linear(in_features=input_dim, out_features=64)

    def forward(self, x):
        x, (h, c) = self.rnn(self.input_fc(x))
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x, (h, c)


class Decoder(nn.Module):
    def __init__(
            self,
            name='Decoder',
            input_dim=32,
            output_dim=32,
            num_layers=1,
            bidirectional=True
    ):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=output_dim // 2 if bidirectional else output_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(in_features=output_dim, out_features=1),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Softmax(dim=-1)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=0.
        )

    def forward(self, x, h, c):
        q = x.permute(1, 0, 2)
        k = x.permute(1, 0, 2)
        v = x.permute(1, 0, 2)

        x, a = self.attn(q, k, v)
        x = x.permute(1, 0, 2)
        x, _ = self.rnn(x, (h, c))
        x = self.fc(x)
        return x


class LSDS(nn.Module):
    def __init__(
            self,
            name='LSDS',
            k=100,
            shapelets_dim=1,
            hidden_dim=32,
            num_classes=1,
            num_layers=1,
            bidirectional=True,
            dropout=0.
    ):
        super(LSDS, self).__init__()
        self.encoder = Encoder(
            input_dim=shapelets_dim,
            output_dim=hidden_dim,
            k=k,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        self.decoder = Decoder(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=64, out_features=16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=16, out_features=num_classes)
        )

        for name, param in self.encoder.named_parameters():
            if name.startswith('rnn.weight'):
                nn.init.orthogonal_(param)
            elif name.startswith('rnn.bias'):
                nn.init.constant_(param, 0)

        for name, param in self.decoder.named_parameters():
            if name.startswith('rnn.weight'):
                nn.init.orthogonal_(param)
            elif name.startswith('rnn.bias'):
                nn.init.constant_(param, 0)

    def forward(self, x):
        x, (h, c) = self.encoder(x)
        p = self.decoder(x, h, c).unsqueeze(dim=-1)
        x = x * p
        x = torch.sum(x, dim=1)
        y = self.classifier(x)
        return y, x, p


class LSDSModel:
    def __init__(self, K, H, dropout=0., lr=0.01, epochs=200, batch_size=16, num_workers=4, num_classes=2, path='./'):
        self.K = K
        self.H = H
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.path = path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = LSDS(
            k=self.K,
            shapelets_dim=1,
            hidden_dim=self.H,
            num_layers=1,
            num_classes=self.num_classes,
            bidirectional=True,
            dropout=self.dropout
        ).to(self.device)

    def train(self, train_x, train_y):
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
        train_loader = DataLoader(
            dataset=TSCDataset(x=train_x, y=train_y),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            dataset=TSCDataset(x=val_x, y=val_y),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
        early_stopping = EarlyStopping(patience=50, verbose=True, path=os.path.join(self.path, 'checkpoint.pt'))

        start_time = time.time()
        for epoch in range(self.epochs):
            train_loss, val_loss = [], []
            acc_metric = torchmetrics.Accuracy(num_classes=self.num_classes).to(self.device)

            self.net.train()
            for _, (x, y) in enumerate(train_loader):
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                y_, _, _ = self.net(x)
                loss = criterion(y_, y)
                acc_metric(y_.argmax(1), y.int())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            train_acc = acc_metric.compute()
            acc_metric.reset()

            self.net.eval()
            with torch.no_grad():
                for _, (x, y) in enumerate(val_loader):
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    y_, _, _ = self.net(x)
                    loss = criterion(y_, y)
                    acc_metric(y_.argmax(1), y.int())

                    val_loss.append(loss.item())

            val_acc = acc_metric.compute()
            acc_metric.reset()
            lr_scheduler.step()

            if epoch % 20 == 0:
                print('learning rate:', lr_scheduler.get_last_lr()[0])

            if (epoch + 1) % 1 == 0:
                print(
                    'Epoch [{}/{}]\t Train Loss: {:.6f}\t Train ACC: {:.6f}\t Val Loss: {:.6f}\t Val ACC: {:.6f}\t'.format(
                        epoch + 1, self.epochs, np.mean(train_loss), train_acc.item(), np.mean(val_loss), val_acc.item()
                    )
                )

            early_stopping(np.mean(val_loss), self.net)
            if early_stopping.early_stop:
                print('Early stopping...')
                break
        self.net = torch.load(os.path.join(self.path, 'checkpoint.pt'))
        end_time = time.time()
        print('time:', end_time - start_time)

    def predict(self, test_x, test_y):
        test_loader = DataLoader(
            dataset=TSCDataset(x=test_x, y=test_y),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        acc_metric = torchmetrics.Accuracy(num_classes=self.num_classes).to(self.device)
        pred = []
        p = []
        self.net.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                y_, _, p_ = self.net(x)
                pred += y_.numpy().tolist()
                p += p_.numpy().tolist()
                acc_metric(y_.argmax(1), y.int())
        test_acc = acc_metric.compute()
        acc_metric.reset()
        return pred, p, test_acc.item()
