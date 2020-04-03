import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet import init
import utils



########################################################################################################################
from cnn.load_datasets import Preprocessing

path = 'dataset/image/CIFAR10'
image_resize = 244


a = Preprocessing()
a.setdata(path, image_resize)
train_data, train_label, test_data, test_label = a.CIFAR10()

batch_size = 32
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.resnet18_v1(classes=10, pretrained=False)

net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=10)
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)