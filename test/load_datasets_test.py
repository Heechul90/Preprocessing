import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet import init
import utils



########################################################################################################################
from cnn.load_datasets import Preprocessing

image_resize = 96
batch_size = 64

a = Preprocessing()
a.setdata(image_resize, batch_size)
train_iter, test_iter = a.MNIST()

for d, l in train_iter:
    break



################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.alexnet(classes=10, pretrained=False)

net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=10)
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)