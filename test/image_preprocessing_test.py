import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet import init
import utils



########################################################################################################################
from cnn.image_preprocessing import Preprocessing

data_path = 'dataset/image/MonkeySpecies/training'
image_resize = 224
test_size = 0.3

a = Preprocessing()
a.setdata(data_path, image_resize, test_size)
train_data, test_data, train_label, test_label = a.image()


batch_size = 32
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(train_data, train_label), batch_size=batch_size)
test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(test_data, test_label), batch_size=batch_size)

for d, l in train_iter:
    break
d.shape
l.shape


################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.alexnet(classes=10, pretrained=False)

net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=1)
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)



########################################################################################################################
from cnn.image_preprocessing import Preprocessing

train_path = 'dataset/image/dogs-vs-cats/training'
test_path = 'dataset/image/dogs-vs-cats/validation'
image_resize = 224
batch_size = 32

a = Preprocessing()
a.setdata(train_path, test_path, image_resize, batch_size)
train_iter, test_iter = a.image()

for d, l in train_iter:
    break



################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.alexnet(classes=1, pretrained=False)

net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=10)
# utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)
