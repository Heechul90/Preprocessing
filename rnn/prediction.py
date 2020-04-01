import pandas as pd
import numpy as np
import mxnet as mx


import time
import glob
import os
import math
import numpy as np
import mxnet as mx
from numpy import array
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.gluon.data import ArrayDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler




# class preprocessing:
#     def __init__(self):
#         self.result = 0
#
#     def add(self, ):
#         a



ctx = mx.cpu(0)
epochs = 30
model_prefix= 'cnnlstm_model'


class timeseries:
def to_supervised(data, label, predict, seq_length):
    X = np.ndarray(shape=(data.shape[0] - (seq_length + predict) + 1, seq_length, data.shape[1]), dtype=np.float32)
    y = np.ndarray(shape=(label.shape[0] - (seq_length + predict) + 1, label.shape[1], label.shape[1]), dtype=np.float32)


    for n in range(data.shape[0]):
        if n + 1 < seq_length:
            continue
        elif n + 2 + predict > data.shape[0]:
            continue
        else:

            x_n = data[n + 1 - seq_length:n + 1, :]        # 20행으로: 20분으로
            y_n = label[n + predict + 1]      # 14분 후 값

        X[n+1 - seq_length] = x_n
        y[n+1 - seq_length] = y_n

    return X, y


def preprocessing(path, seq_length, predict, split, batch_size):
    dataset = pd.read_csv(path, index_col=None, header=None)

    data = dataset.values.astype(np.float32)
    label = dataset.iloc[:, -1:].values.astype(np.float32)

    data = data.reshape(data.shape[0], len(data[0]))
    label = label.reshape(label.shape[0], len(label[0]))

    scaler_ = StandardScaler()
    data = scaler_.fit_transform(data)
    scaler_ = StandardScaler()
    label = scaler_.fit_transform(label)

    data, label = to_supervised(data, label, predict, seq_length)

    train_data = data[:int(len(data)*split)]
    test_data = data[int(len(data)*split):]
    train_label = label[:int(len(label)*split)]
    test_label = label[int(len(label)*split):]

    train_set = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
    test_set = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

    return train_set, test_set

path = 'dataset/test.csv'
seq_length = 14
predict = 7
split = 0.8
batch_size = 32

train_iter, test_iter = preprocessing(path, seq_length, predict, split, batch_size)

for d, l in train_iter:
    break
print(d.shape, l.shape)
########################################################################################################################
model = mx.gluon.nn.Sequential()

with model.name_scope():
    model.add(mx.gluon.rnn.LSTM(128))
    model.add(mx.gluon.nn.Dense(1, activation='tanh'))


### Training & Evaluation
# loss 함수 선택
L = gluon.loss.L2Loss() # L2 loss: (실제값 - 예측치)제곱해서 더한 값, L1 loss: (실제값 - 예측치)절대값해서 더한 값

# 평가
def evaluate_accuracy(data_iterator, model, L):
    loss_avg = 0.
    for i, (d, l) in enumerate(data_iterator):
        data = d.as_in_context(ctx).reshape((-1, 1, 1))
        label = l.as_in_context(ctx).reshape((-1, 1, 1))
        output = model(data)
        loss = L(output, data)
        loss_avg = (loss_avg * i + nd.mean(loss).asscalar()) / (i + 1)
    return loss_avg



# cpu or gpu
ctx = mx.cpu()


# Xavier는 모든 layers에서 gradient scale이 거의 동일하게 유지하도록 설계됨
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# 이 모델에는 sgd 최적화 함수, 학습률은 0.01이 가장 최적으로 보임
# sgd에 비해 너무 작지 않고, 최적화에 시간이 오래 걸리지 않으며 너무 크지 않는다
# 그래서 loss function의 최소값을 넘지 않는다
trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01})


### Let’s run the training loop and plot MSEs
epochs = 30
training_mse = []        # 평균 제곱 오차를 기록
validation_mse = []

for epoch in range(epochs):
    print(str(epoch+1))
    for i, (d, l) in enumerate(train_iter):
        data = d.as_in_context(ctx).reshape((-1, 1, 1))
        label = l.as_in_context(ctx).reshape((-1, 1, 1))

        with autograd.record():
            output = model(data)
            loss = L(output, data)

        loss.backward()
        trainer.step(batch_size)

    training_mse.append(evaluate_accuracy(train_iter, model, L))
    validation_mse.append(evaluate_accuracy(test_iter, model, L))

plt.plot(training_mse, color='r')
plt.plot(validation_mse, color='b')