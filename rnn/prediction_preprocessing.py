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
    def setdata(self, path, seq_length, predict, split, batch_size):
        self.path = path
        self.seq_length = seq_length
        self.predict = predict
        self.split = split
        self.batch_size = batch_size


    def prediction(self):
        path, seq_length, predict, split, batch_size = self.path, self.seq_length, self.predict, self.split, self.batch_size

        # csv파일을 읽음
        dataset = pd.read_csv(path, index_col=None, header=None)

        # 데이터를 array형태로 변환
        data = dataset.values.astype(np.float32)
        label = dataset.iloc[:, -1:].values.astype(np.float32)

        # 데이터 reshape
        data = data.reshape(data.shape[0], len(data[0]))
        label = label.reshape(label.shape[0], len(label[0]))

        # 데이터 정규화(정규화, 표준화등 기능을 추가 할 수 있음)
        # scaler_ = StandardScaler()
        # data = scaler_.fit_transform(data)
        # scaler_ = StandardScaler()
        # label = scaler_.fit_transform(label)

        # sequence 입력값과 타겟값 설정
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

        # 데이터를 trainset과 testset으로 구분
        train_data = data[:int(len(data) * split)]
        test_data = data[int(len(data) * split):]
        train_label = label[:int(len(label) * split)]
        test_label = label[int(len(label) * split):]

        # 데이터를 batch로 불러옴
        train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
        test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

        return train_iter, test_iter


