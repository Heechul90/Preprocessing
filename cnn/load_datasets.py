import mxnet as mx
from mxnet import gluon, autograd, nd
import numpy as np
from sklearn.model_selection import train_test_split


class Preprocessing():
    def setdata(self, image_resize, batch_size):
        self.image_resize = image_resize
        self.batch_size = batch_size

    def MNIST(self):
        image_resize, batch_size = self.image_resize, self.batch_size

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.MNIST('dataset/image/MNIST', train=True, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        test_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.MNIST('dataset/image/MNIST', train=False, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        return train_iter, test_iter

    def FashionMNIST(self):
        image_resize, batch_size = self.image_resize, self.batch_size

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.FashionMNIST('dataset/image/FashionMNIST', train=True, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        test_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.FashionMNIST('dataset/image/FashionMNIST', train=False, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        return train_iter, test_iter

    def CIFAR10(self):
        image_resize, batch_size = self.image_resize, self.batch_size

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.CIFAR10('dataset/image/CIFAR10', train=True, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        test_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.CIFAR10('dataset/image/CIFAR10', train=False, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        return train_iter, test_iter

    def CIFAR100(self):
        image_resize, batch_size = self.image_resize, self.batch_size

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        train_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.CIFAR100('dataset/image/CIFAR100', train=True, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        test_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.CIFAR100('dataset/image/CIFAR100', train=False, transform=transformer),
            batch_size=batch_size, shuffle=False, last_batch='discard')

        return train_iter, test_iter

