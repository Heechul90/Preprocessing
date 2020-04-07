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

for d, l in train_iter:
    break
print(l[1])

### model
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(32, activation="relu"),
            nn.Dense(64, activation="relu"),
            nn.Dense(1))

# ### train
# ctx = utils.try_gpu()
# net.initialize(ctx=ctx)
# print('initialize weight on', ctx)
#
# loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
# # utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)
# utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)


import sys
import os
import json
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from time import time
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from gluoncv.data.transforms.presets.imagenet import transform_eval

def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        predictions = nd.argmax(net(data.as_in_context(ctx)), axis=1)
        acc.update(preds=predictions, labels=label.as_in_context(ctx))
    return acc.get()[1]




mx.random.seed(1)
epochs = 50
lr = 0.1
num_workers = 0
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()
ctx=mx.cpu()

# Initialize parameters randomly
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

train_start = time()
all_train_mse = []
all_test_mse = []
test_label = []
train_imgs = []
scores_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # Wait for completion of previous iteration to
        # avoid unnecessary memory allocation
        nd.waitall()
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        metric.update([label], [output])
        if i % 10 == 0 and i > 0:
            name, acc = metric.get()
            print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
            sys.stdout.flush()

    train_mse = evaluate_accuracy(train_iter, net, ctx)
    test_mse = evaluate_accuracy(test_iter, net, ctx)
    all_train_mse.append(train_mse)
    all_test_mse.append(test_mse)

    name, acc = metric.get()
    print('[Epoch %d] Training: %s=%f' % (e, name, acc))
    sys.stdout.flush()

save_model("model_trained")

eval_start = time()
print("{} workers: train duration {:.4}".format(
    num_workers, eval_start - train_start))
sys.stdout.flush()

plt.plot(all_train_mse)
plt.plot(all_test_mse)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend(['train', 'valid'])
plt.savefig('accuracy.jpg')

for i, (data, label) in enumerate(test_iter):

    if i % 10 == 0 and i > 0:
        name, acc = metric.get()
        print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
        predictions = nd.argmax(net(data.as_in_context(ctx)), axis=1)
        test_label.append(label)
        scores_test.append(predictions)

test_label = nd.concat(*test_label, dim=0)
scores_test = nd.concat(*scores_test, dim=0)
# print(test_label)
# print("----------------")
# print(scores_test)
max = np.max(test_label.asnumpy())

myClasses = list(range(0, int(max + 1), 1))
label = label_binarize(test_label.asnumpy(), classes=myClasses)
pred = label_binarize(scores_test.asnumpy(), classes=myClasses)
plt.clf()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(int(max + 1)):
    fpr[i], tpr[i], _ = metrics.roc_curve(label[:, i], pred[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('roc_curve.jpg')

# precision
precision = precision_score(y_true=test_label.asnumpy(), y_pred=scores_test.asnumpy(),
                            average='weighted')  # , zero_division=0)
print('Precision: %f' % precision)
# recall
recall = recall_score(y_true=test_label.asnumpy(), y_pred=scores_test.asnumpy(), average='weighted')
print('Recall: %f' % recall)
# f1_score
f1 = f1_score(y_true=test_label.asnumpy(), y_pred=scores_test.asnumpy(), average='weighted')
print('F1 score: %f' % f1)

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

for d, l in train_iter:
    break
print(l[0])

### model
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(32, activation="relu"),
            nn.Dense(64, activation="relu"),
            nn.Dense(3))

# ### train
# ctx = utils.try_gpu()
# net.initialize(ctx=ctx)
# print('initialize weight on', ctx)
#
#
#
# loss = gluon.loss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
# utils.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=50)



import sys
import os
import json
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from time import time
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from gluoncv.data.transforms.presets.imagenet import transform_eval

def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        predictions = nd.argmax(net(data.as_in_context(ctx)), axis=1)
        acc.update(preds=predictions, labels=label.as_in_context(ctx))
    return acc.get()[1]




mx.random.seed(1)
epochs = 50
lr = 0.1
num_workers = 0
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()
ctx=mx.cpu()

# Initialize parameters randomly
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

train_start = time()
all_train_mse = []
all_test_mse = []
test_label = []
train_imgs = []
scores_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # Wait for completion of previous iteration to
        # avoid unnecessary memory allocation
        nd.waitall()
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        metric.update([label], [output])
        if i % 10 == 0 and i > 0:
            name, acc = metric.get()
            print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
            sys.stdout.flush()

    train_mse = evaluate_accuracy(train_iter, net, ctx)
    test_mse = evaluate_accuracy(test_iter, net, ctx)
    all_train_mse.append(train_mse)
    all_test_mse.append(test_mse)

    name, acc = metric.get()
    print('[Epoch %d] Training: %s=%f' % (e, name, acc))
    sys.stdout.flush()

save_model("model_trained")

eval_start = time()
print("{} workers: train duration {:.4}".format(
    num_workers, eval_start - train_start))
sys.stdout.flush()

plt.plot(all_train_mse)
plt.plot(all_test_mse)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.legend(['train', 'valid'])
plt.savefig('accuracy.jpg')

for i, (data, label) in enumerate(test_iter):

    if i % 10 == 0 and i > 0:
        name, acc = metric.get()
        print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
        predictions = nd.argmax(net(data.as_in_context(ctx)), axis=1)
        test_label.append(label)
        scores_test.append(predictions)

test_label = nd.concat(*test_label, dim=0)
scores_test = nd.concat(*scores_test, dim=0)
# print(test_label)
# print("----------------")
# print(scores_test)
max = np.max(test_label.asnumpy())

myClasses = list(range(0, int(max + 1), 1))
label = label_binarize(test_label.asnumpy(), classes=myClasses)
pred = label_binarize(scores_test.asnumpy(), classes=myClasses)
plt.clf()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(int(max + 1)):
    fpr[i], tpr[i], _ = metrics.roc_curve(label[:, i], pred[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('roc_curve.jpg')

# precision
precision = precision_score(y_true=test_label.asnumpy(), y_pred=scores_test.asnumpy(),
                            average='weighted')  # , zero_division=0)
print('Precision: %f' % precision)
# recall
recall = recall_score(y_true=test_label.asnumpy(), y_pred=scores_test.asnumpy(), average='weighted')
print('Recall: %f' % recall)
# f1_score
f1 = f1_score(y_true=test_label.asnumpy(), y_pred=scores_test.asnumpy(), average='weighted')
print('F1 score: %f' % f1)
########################################################################################################################
