import mxnet as mx
from mxnet import gluon, autograd, nd
import numpy as np
from sklearn.model_selection import train_test_split


class Preprocessing():
    def setdata(self, path, image_resize):
        self.path = path
        self.image_resize = image_resize

    def MNIST(self):
        path, image_resize = self.path, self.image_resize

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.vision.datasets.MNIST(path, train=True, transform=transformer)
        test_iter = gluon.data.vision.datasets.MNIST(path, train=False, transform=transformer)

        train_data = []
        train_label = []
        for train_d, train_l in train_iter:
            train_data.append(train_d)
            train_label.append(train_l)

        test_data = []
        test_label = []
        for test_d, test_l in test_iter:
            test_data.append(test_d)
            test_label.append(test_l)

        return train_data, train_label, test_data, test_label

    def FashionMNIST(self):
        path, image_resize = self.path, self.image_resize

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.vision.datasets.FashionMNIST(path, train=True, transform=transformer)
        test_iter = gluon.data.vision.datasets.FashionMNIST(path, train=False, transform=transformer)

        train_data = []
        train_label = []
        for train_d, train_l in train_iter:
            train_data.append(train_d)
            train_label.append(train_l)

        test_data = []
        test_label = []
        for test_d, test_l in test_iter:
            test_data.append(test_d)
            test_label.append(test_l)

        return train_data, train_label, test_data, test_label

    def CIFAR10(self):
        path, image_resize = self.path, self.image_resize

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.vision.datasets.CIFAR10(path, train=True, transform=transformer)
        test_iter = gluon.data.vision.datasets.CIFAR10(path, train=False, transform=transformer)

        train_data = []
        train_label = []
        for train_d, train_l in train_iter:
            train_data.append(train_d)
            train_label.append(train_l)

        test_data = []
        test_label = []
        for test_d, test_l in test_iter:
            test_data.append(test_d)
            test_label.append(test_l)

        return train_data, train_label, test_data, test_label

    def CIFAR100(self):
        path, image_resize = self.path, self.image_resize

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.vision.datasets.CIFAR100(path, train=True, transform=transformer)
        test_iter = gluon.data.vision.datasets.CIFAR100(path, train=False, transform=transformer)

        train_data = []
        train_label = []
        for train_d, train_l in train_iter:
            train_data.append(train_d)
            train_label.append(train_l)

        test_data = []
        test_label = []
        for test_d, test_l in test_iter:
            test_data.append(test_d)
            test_label.append(test_l)

        return train_data, train_label, test_data, test_label
