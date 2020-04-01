import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn import preprocessing

class Preprocessing():
    def setdata(self, data_path, val):
        self.data_path = data_path
        self.val = val

    def autoencoder(self):
        path, validation = self.data_path, self.val
        data = pd.read_csv(path, header=None, sep=',')

        train_data = data[: int(len(data) * validation)]
        validation_data = data[int(len(data) * validation): ]

        return train_data, validation_data


