import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import utils


########################################################################################################################
from timeseries.text_preprocessing import Preprocessing

path = 'dataset/text/text_test.csv'
validation = 0.8
batch_size = 32

a = Preprocessing()
a.setdata(path, validation, batch_size)
train_iter, test_iter = a.text_generation()




########################################################################################################################
import pandas as pd

df = pd.read_csv('dataset/text/ArticlesApril2018.csv')
df = df['headline']
df.to_csv('dataset/text/text_test.csv', index=None, header=None, encoding='utf-8')
df1 = pd.read_csv(path, header=None, encoding='utf-8')