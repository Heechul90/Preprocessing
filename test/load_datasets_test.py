import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet import init
import utils


########################################################################################################################
##### MNIST
from cnn.load_datasets import Preprocessing

path = 'dataset/image/MNIST'
image_resize = 244
batch_size = 32

a = Preprocessing()
a.setdata(path, image_resize, batch_size)
train_iter, test_iter = a.MNIST()

################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.resnet18_v1(classes=10, pretrained=False)

net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs = 10)
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)


########################################################################################################################
##### FashionMNIST
from cnn.load_datasets import Preprocessing

path = 'dataset/image/FashionMNIST'
image_resize = 244
batch_size = 32

a = Preprocessing()
a.setdata(path, image_resize, batch_size)
train_iter, test_iter = a.FashionMNIST()

################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.resnet18_v1(classes=10, pretrained=False)

net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs = 10)
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)


########################################################################################################################
# from cnn.load_datasets import Preprocessing
#
# path = 'dataset/image/CIFAR10'
# image_resize = 244
# batch_size = 32
#
# a = Preprocessing()
# a.setdata(path, image_resize, batch_size)
# train_iter, test_iter = a.CIFAR10()
#
# ################## model
# from mxnet.gluon.model_zoo import vision
# ctx = mx.cpu()
# net = vision.resnet18_v1(classes=10, pretrained=False)
#
# net.initialize(ctx=ctx, init=init.Xavier())
#
# loss = gluon.loss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
# utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs = 10)
# # utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)


########################################################################################################################
from cnn.load_datasets import Preprocessing

path = 'dataset/image/CIFAR100'
image_resize = 244
batch_size = 32

a = Preprocessing()
a.setdata(path, image_resize, batch_size)
train_iter, test_iter = a.CIFAR100()

################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.resnet18_v1(classes=10, pretrained=False)

net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs = 10)
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)