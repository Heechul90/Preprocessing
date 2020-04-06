import pandas as pd
import numpy as np
from numpy import split
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

path = 'dataset1/iris.csv'
test_size = 0.2
batch_size = 32

class Preprocessing():
    def setdata(self, path, test_size, batch_size):
        self.path = path
        self.test_size = test_size
        self.batch_size = batch_size

    def label(self):
        path, test_size, batch_size = self.path, self.test_size, self.batch_size

        # 데이터 불러오기
        df = pd.read_csv(path, header=None)

        # 읽어온 데이터 array로 변환
        df = df.values.astype('float32')

        # 입력값과 출력값 나누기
        X = df[:,:-1]     # 마지막 컬럼을 제외한 나머지 컬럼
        y = df[:,-1]      # 출력값(타겟값)인 마지막 컬럼

        # test_size 입력하여 test를 몇으로 할건지, shuffle값이 default면 True
        train_data, test_data, train_label, test_label = train_test_split(X, y, test_size = test_size, shuffle=False)

        # DataLoader를 이용하여 batch_size 결정
        train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size, shuffle=False)
        test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size, shuffle=False)

        return train_iter, test_iter

    def nolabel(self):
        path, test_size, batch_size = self.path, self.test_size, self.batch_size

        # 데이터 불러오기
        df = pd.read_csv(path, header=None)

        # df = df.sample(frac=1)
        df = df.values

        # 입력값과 출력값 나누기
        X = df[:,:-1].astype('float32') # 마지막 컬럼을 제외한 나머지 컬럼
        y_obj = df[:,-1]                # 출력값(타겟값)인 마지막 컬럼

        # 문자로 된 label 숫자로 encoder
        e = LabelEncoder()
        e.fit(y_obj)
        y = e.transform(y_obj)
        y = y.astype('float32')

        # test_size 입력하여 test를 몇으로 할건지, shuffle값이 default면 True
        train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=test_size, shuffle=False)

        # DataLoader를 이용하여 batch_size 결정
        train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size, shuffle=False)
        test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size, shuffle=False)

        return train_iter, test_iter
########################################################################################################################
