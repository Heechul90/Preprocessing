### dense_preprocessing.py test

import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import utils

########################################################################################################################
### 라벨링 2중 분류 문제
from cnn.dense_preprocessing import Preprocessing

path = 'dataset1/pima-indians-diabetes.csv'
test_size = 0.8
batch_size = 32

a = Preprocessing()
a.setdata(path, test_size, batch_size)
train_iter, test_iter = a.label()

### model
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(32, activation="relu"),
            nn.Dense(64, activation="relu"),
            nn.Dense(1))

### train
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
print('initialize weight on', ctx)

loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
# utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)


########################################################################################################################
### 노라벨 다중 분류 문제
from cnn.dense_preprocessing import Preprocessing

### data
path = 'dataset1/iris.csv'
test_size = 0.2
batch_size = 32

a = Preprocessing()
a.setdata(path, test_size, batch_size)
train_iter, test_iter = a.nolabel()

### model
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(32, activation="relu"),
            nn.Dense(64, activation="relu"),
            nn.Dense(3))

### train
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
print('initialize weight on', ctx)



loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)

########################################################################################################################
