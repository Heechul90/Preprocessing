import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessing():
    def setdata(self, data_path, val):
        self.data_path = data_path
        self.val = val

    def autoencoder(self):
        path, validation = self.data_path, self.val
        data = pd.read_csv(path, header=None, sep=',')

        data = data.values.astype(np.float32)

        # scaler_ = StandardScaler()
        # data = scaler_.fit_transform(data)

        train_data = data[: int(len(data) * validation)]
        validation_data = data[int(len(data) * validation): ]

        return train_data, validation_data


