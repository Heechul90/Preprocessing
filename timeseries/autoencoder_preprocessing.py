import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessing():
    def setdata(self, train_path, test_path, val):
        self.train_path = train_path
        self.test_path = test_path
        self.val = val

    def autoencoder(self):
        train_path, test_path, validation = self.train_path, self.test_path, self.val
        train_data = pd.read_csv(train_path, header=None, sep=',')
        test_data = pd.read_csv(test_path, header=None, sep=',')

        train_data = train_data.values.astype(np.float32)
        test_data = test_data.values.astype(np.float32)

        # scaler_ = StandardScaler()
        # train_data = scaler_.fit_transform(train_data)
        # test_data = scaler_.fit_transform(test_data)

        train_data = train_data[: int(len(train_data) * validation)]
        validation_data = train_data[int(len(train_data) * validation): ]

        return train_data, test_data, validation_data


