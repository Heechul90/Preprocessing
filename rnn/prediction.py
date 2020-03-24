import pandas as pd
import numpy as np
import mxnet as mx


import time
import glob
import os
import math
import numpy as np
import mxnet as mx
from numpy import array
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.gluon.data import ArrayDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler




# class preprocessing:
#     def __init__(self):
#         self.result = 0
#
#     def add(self, ):
#         a



ctx = mx.cpu(0)
epochs = 30
model_prefix= 'cnnlstm_model'



def to_supervised(data, label, predict, seq_length):
    X = np.ndarray(shape=(data.shape[0] - (seq_length + predict) + 1, seq_length, data.shape[1]), dtype=np.float32)
    y = np.ndarray(shape=(label.shape[0] - (seq_length + predict) + 1, label.shape[1], label.shape[1]), dtype=np.float32)


    for n in range(data.shape[0]):
        if n + 1 < seq_length:
            continue
        elif n + 2 + predict > data.shape[0]:
            continue
        else:

            x_n = data[n + 1 - seq_length:n + 1, :]        # 20행으로: 20분으로
            y_n = label[n + predict + 1]      # 14분 후 값

        X[n+1 - seq_length] = x_n
        y[n+1 - seq_length] = y_n

    return X, y


def preprocessing(path, seq_length, predict, split, batch_size):
    dataset = pd.read_csv(path, index_col=None, header=None)

    data = dataset.values.astype(np.float32)
    label = dataset.iloc[:, -1:].values.astype(np.float32)

    data = data.reshape(data.shape[0], len(data[0]))
    label = label.reshape(label.shape[0], len(label[0]))

    scaler_ = StandardScaler()
    data = scaler_.fit_transform(data)
    scaler_ = StandardScaler()
    label = scaler_.fit_transform(label)

    data, label = to_supervised(data, label, predict, seq_length)

    train_data = data[:int(len(data)*split)]
    test_data = data[int(len(data)*split):]
    train_label = label[:int(len(label)*split)]
    test_label = label[int(len(label)*split):]

    train_set = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
    test_set = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

    return train_set, test_set

path = 'dataset/test.csv'
seq_length = 60
predict = 30
split = 0.8
batch_size=64

train_iter, test_iter = preprocessing(path, seq_length, predict, split, batch_size)

for d, l in train_iter:
    break
print(d.shape, l.shape)
########################################################################################################################
