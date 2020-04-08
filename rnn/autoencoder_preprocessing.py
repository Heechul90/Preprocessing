import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessing():
    def setdata(self, path, split, batch_size):
        self.path = path
        self.split = split
        self.batch_size = batch_size

    def autoencoder(self):
        path, split, batch_size = self.path, self.split, self.batch_size

        # csv 파일 읽기
        data = pd.read_csv(path, header=None, sep=',')

        # 데이터를 array형태로 변환
        data = data.values.astype(np.float32)

        # trainset과 testset으로 구분
        train_data = data[: int(len(data) * split)]
        test_data = data[int(len(data) * split): ]

        # DataLoader를 이용해서 데이터를 batch로 불러옴
        train_data = mx.gluon.data.DataLoader(train_data, batch_size, shuffle=False)
        test_data = mx.gluon.data.DataLoader(test_data, batch_size, shuffle=False)

        return train_data, test_data


