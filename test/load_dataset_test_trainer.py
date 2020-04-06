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

path = 'dataset/image/MonkeySpecies2'
image_resize = 96
batch_size = 32

a = Preprocessing()
a.setdata(path, image_resize, batch_size)
train_iter, test_iter = a.MNIST()


########################################################################################################################
from cnn.image_preprocessing import Preprocessing

path = 'dataset/image/MonkeySpecies2'
image_resize = 96
batch_size = 32

a = Preprocessing()
a.setdata(path, image_resize, batch_size)
train_iter, test_iter = a.MNIST()

################## model
from mxnet.gluon.model_zoo import vision
ctx = mx.cpu()
net = vision.resnet18_v1(classes=10, pretrained=False)





########################################################################################################################
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
epochs = 1
lr = 0.1
num_workers = 0
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()

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