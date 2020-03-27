import pandas as pd
import numpy as np
import mxnet as mx
from numpy import array
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.gluon.data import ArrayDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


class Preprocessing():
    def setdata(self, data_path, seq, pred, sp):
        self.data_path = data_path
        self.seq = seq
        self.pred = pred
        self.sp = sp


    def preprocessing(self):
        path, seq_length, predict, split = self.data_path, self.seq, self.pred, self.sp

        dataset = pd.read_csv(path, index_col=None, header=None)

        data = dataset.values.astype(np.float32)
        label = dataset.iloc[:, -1:].values.astype(np.float32)

        data = data.reshape(data.shape[0], len(data[0]))
        label = label.reshape(label.shape[0], len(label[0]))

        scaler_ = StandardScaler()
        data = scaler_.fit_transform(data)
        scaler_ = StandardScaler()
        label = scaler_.fit_transform(label)

        def to_supervised(data, label, predict, seq_length):
            X = np.ndarray(shape=(data.shape[0] - (seq_length + predict) + 1, seq_length, data.shape[1]),
                           dtype=np.float32)
            y = np.ndarray(shape=(label.shape[0] - (seq_length + predict) + 1, label.shape[1], label.shape[1]),
                           dtype=np.float32)

            for n in range(data.shape[0]):
                if n + 1 < seq_length:
                    continue
                elif n + 2 + predict > data.shape[0]:
                    continue
                else:
                    x_n = data[n + 1 - seq_length:n + 1, :]  # 20행으로: 20분으로
                    y_n = label[n + predict + 1]  # 14분 후 값

                X[n + 1 - seq_length] = x_n
                y[n + 1 - seq_length] = y_n

            return X, y

        data, label = to_supervised(data, label, predict, seq_length)

        train_data = data[:int(len(data) * split)]
        test_data = data[int(len(data) * split):]
        train_label = label[:int(len(label) * split)]
        test_label = label[int(len(label) * split):]

        # train_set = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
        # test_set = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

        return train_data, test_data, train_label, test_label


