# -*- coding: utf-8 -*-
from timeseries.text_preprocessing import Preprocessing
import mxnet as mx
from mxnet import nd, autograd, gluon

path = 'dataset/text/text_test.csv'
validation = 0.8

a = Preprocessing()
a.setdata(path, validation)
train_data, validation_data, train_label, validation_label = a.text_generation()

len(train_data)
len(validation_data)



batch_size=32
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(validation_data, validation_label), batch_size=batch_size)



