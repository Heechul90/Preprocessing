import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessing():
    def setdata(self, path, split):
        self.path = path
        self.split = split

    def autoencoder(self):
        path, split = self.path, self.split

        path = 'dataset/autoencoder/train_test.csv'
        data = pd.read_csv(path, header=None, sep=',')

        data = data.values.astype(np.float32)

        train_data = data[: int(len(data) * split)]
        test_data = data[int(len(data) * split): ]

        return train_data, test_data


