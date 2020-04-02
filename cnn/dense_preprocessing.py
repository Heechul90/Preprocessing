import pandas as pd
import numpy as np
from numpy import split
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Preprocessing():
    def setdata(self, data_path, sp):
        self.data_path = data_path
        self.sp = sp
        # self.batch = batch

    def label(self):
        path, split = self.data_path, self.sp

        df = pd.read_csv(path, header=None)
        # df = df.sample(frac=1)
        df = df.values.astype('float32')

        X = df[:,:-1]
        y = df[:,-1]

        train_data = X[:int(len(X) * split)]
        test_data = X[int(len(X) * split):]
        train_label = y[:int(len(y) * split)]
        test_label = y[int(len(y) * split):]

        # train_data, test_data, train_label, test_label = train_test_split(X, y, test_size = (1 - split))

        # train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
        # test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

        return train_data, test_data, train_label, test_label


    def nolabel(self):
        path, split = self.data_path, self.sp

        df = pd.read_csv(path, header=None)
        # df = df.sample(frac=1)
        df = df.values
        X = df[:,:-1].astype('float32')
        y_obj = df[:,-1]

        e = LabelEncoder()
        e.fit(y_obj)
        y = e.transform(y_obj)
        y = y.astype('float32')

        train_data = X[:int(len(X) * split)]
        test_data = X[int(len(X) * split):]
        train_label = y[:int(len(y) * split)]
        test_label = y[int(len(y) * split):]

        # train_data, test_data, train_label, test_label = train_test_split(X, y, test_size = (1 - split))

        # train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
        # test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

        return train_data, test_data, train_label, test_label
########################################################################################################################
