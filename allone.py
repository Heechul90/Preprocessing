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

import DataBlockClass


class aoTrainingCls:
    def __init__(self, Processor, alloneLogger):
        self.Logger = alloneLogger
        if Processor == "cpu":
            self.Logger.debug("CPU 모드로 동작")
            self.ctx = mx.cpu()  # [mx.cpu()] # using a cpu
        else:
            self.Logger.debug("GPU 모드로 동작")
            self.ctx = mx.gpu(0)  # [mx.gpu(i) for i in mx.test_utils.list_gpus()] # using gpus

        self.jsonfile = "model-symbol.json"
        self.paramfile = "model-0000.params"
        self.datafiles = ['data']

    def setParam(self, lr, epochs, momentum, batch_size, kvValue, optimizer):
        self.lr = lr  # 0.05 # learning rate
        self.epochs = epochs  # 3 # epochs 10
        self.momentum = momentum  # 0.9 # momentum
        self.batch_size = batch_size  # 3 # batchsize 100
        self.kv = mx.kv.create(kvValue)  # "local", "dist"
        self.optimizer = optimizer  # 'sgd'
        self.kvValue = kvValue

    def evaluate_accuracy(self, data_iterator, net, ctx):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            predictions = nd.argmax(net(data.as_in_context(ctx)), axis=1)
            acc.update(preds=predictions, labels=label.as_in_context(ctx))
        return acc.get()[1]

    def exeMNISTTrain(self, net, aoObjData):
        num_workers = 0
        self.train_data = aoObjData.data_loaderCIFAR10(True, self.batch_size, num_workers)
        self.test_data = aoObjData.data_loaderCIFAR10(False, self.batch_size, num_workers)

        self.Trainer(net)

    def exeMNISTTrainDist(self, net, aoObjData):
        num_workers = self.kv.num_workers
        self.train_data = aoObjData.data_loaderMNIST(True, self.batch_size, num_workers)
        self.test_data = aoObjData.data_loaderMNIST(False, self.batch_size, num_workers)

        self.TrainerDist(net)

    def Trainer(self, net):
        mx.random.seed(1)
        epochs = self.epochs
        num_workers = 0
        self.net = net
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        metric = mx.metric.Accuracy()

        # Initialize parameters randomly
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx, force_reinit=True)
        trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': self.lr})

        train_start = time()
        all_train_mse = []
        all_test_mse = []
        test_label = []
        train_imgs = []
        scores_test = []

        for e in range(self.epochs):
            for i, (data, label) in enumerate(self.train_data):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                # Wait for completion of previous iteration to
                # avoid unnecessary memory allocation
                nd.waitall()
                with autograd.record():
                    output = self.net(data)
                    loss = softmax_cross_entropy(output, label)
                loss.backward()
                trainer.step(data.shape[0])
                metric.update([label], [output])
                if i % 10 == 0 and i > 0:
                    name, acc = metric.get()
                    print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
                    sys.stdout.flush()

            train_mse = self.evaluate_accuracy(self.train_data, self.net, self.ctx)
            test_mse = self.evaluate_accuracy(self.test_data, self.net, self.ctx)
            all_train_mse.append(train_mse)
            all_test_mse.append(test_mse)

            name, acc = metric.get()
            print('[Epoch %d] Training: %s=%f' % (e, name, acc))
            sys.stdout.flush()

        self.save_model("model_trained")

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

        for i, (data, label) in enumerate(self.test_data):

            if i % 10 == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
                predictions = nd.argmax(self.net(data.as_in_context(self.ctx)), axis=1)
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

    def TrainerDist(self, net):
        mx.random.seed(1)
        epochs = self.epochs
        # self.ctx = mx.cpu()
        self.net = net
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        metric = mx.metric.Accuracy()

        # Initialize parameters randomly
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx, force_reinit=True)
        # multi server
        trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': self.lr}, kvstore=self.kv,
                                update_on_kvstore=True)

        stride = int(self.batch_size / self.kv.num_workers)
        start = self.kv.rank * stride
        end = start + stride
        if self.kv.rank == self.kv.num_workers:
            end = self.batch_size

        train_start = time()
        all_train_mse = []
        all_test_mse = []

        for e in range(self.epochs):
            for i, (data, label) in enumerate(self.train_data):
                data = data[start:end].as_in_context(self.ctx)
                label = label[start:end].as_in_context(self.ctx)
                # Wait for completion of previous iteration to
                # avoid unnecessary memory allocation
                nd.waitall()
                with autograd.record():
                    output = self.net(data)
                    loss = softmax_cross_entropy(output, label)
                loss.backward()
                trainer.step(data.shape[0])
                metric.update([label], [output])
                if i % 10 == 0 and i > 0:
                    name, acc = metric.get()
                    print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
                    sys.stdout.flush()

            train_mse = self.evaluate_accuracy(self.train_data, self.net, self.ctx)
            test_mse = self.evaluate_accuracy(self.test_data, self.net, self.ctx)
            all_train_mse.append(train_mse)
            all_test_mse.append(test_mse)

            name, acc = metric.get()
            print('[Epoch %d] Training: %s=%f' % (e, name, acc))
            sys.stdout.flush()

        self.save_model("model_trained")

        eval_start = time()
        print("{} workers: train duration {:.4}".format(
            num_workers, eval_start - train_start))
        sys.stdout.flush()

        plt.plot(all_train_mse)
        plt.plot(all_test_mse)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Result')
        plt.legend(['train', 'valid'])
        plt.savefig('accuracy.jpg')

        for i, (data, label) in enumerate(self.test_data):

            if i % 100 == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f' % (e, i, name, acc))
                predictions = nd.argmax(self.net(data.as_in_context(self.ctx)), axis=1)
                train_label.append(label)
                scores_train.append(predictions)

        train_label = nd.concat(*train_label, dim=0)
        scores_train = nd.concat(*scores_train, dim=0)
        max = np.max(train_label.asnumpy())

        myClasses = list(range(0, int(max + 1), 1))
        label = label_binarize(train_label.asnumpy(), classes=myClasses)
        pred = label_binarize(scores_train.asnumpy(), classes=myClasses)
        plt.clf()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(int(max + 1)):
            fpr[i], tpr[i], _ = metrics.roc_curve(label[:, i], pred[:, i])
            plt.plot(fpr[i], tpr[i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        plt.clf()
        plt.plot([0, 1], [0, 1], 'k--')
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

    def save_model(self, filename):
        self.net.save_parameters(filename)

