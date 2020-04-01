
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import utils



########################################################################################################################
from cnn.dense_preprocessing import Preprocessing

path = 'dataset1/pima-indians-diabetes.csv'
split = 0.8
batch_size = 32

a = Preprocessing()
# a.setdata(path, split)
# train_iter, test_iter = a.label()
a.setdata(path, split)
train_data, test_data, train_label, test_label = a.label()

### model
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(32, activation="relu"),
            nn.Dense(64, activation="relu"),
            nn.Dense(128, activation="relu"),
            nn.Dense(256, activation="relu"),
            nn.Dense(1))

### train
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
print('initialize weight on', ctx)

loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
# utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)
utils.train(train_data, test_data, train_label, test_label, batch_size, net, loss, trainer, ctx, num_epochs=50)



########################################################################################################################
from cnn.dense_preprocessing import Preprocessing

### data
path = 'dataset1/iris.csv'
split = 0.8
batch_size = 32

a = Preprocessing()
# a.setdata(path, split)
# train_iter, test_iter = a.label()
a.setdata(path, split)
train_data, test_data, train_label, test_label = a.nolabel()

train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

t = []
a = []
for d, l in test_iter:
    t.append(d)
    a.append(l)


### model
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(32, activation="relu"),
            nn.Dense(64, activation="relu"),
            nn.Dense(128, activation="relu"),
            nn.Dense(256, activation="relu"),
            nn.Dense(3))

### train
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
print('initialize weight on', ctx)



loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)
# utils.train(train_data, test_data, train_label, test_label, batch_size, net, loss, trainer, ctx, num_epochs=50)
utils.score(train_data, test_data, train_label, test_label)