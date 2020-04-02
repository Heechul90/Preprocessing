import pandas as pd
import numpy as np
from numpy import split
import mxnet as mx
from mxnet import gluon, autograd, nd

train_path = 'dataset/image/MonkeySpecies/training'
test_path = 'dataset/image/MonkeySpecies/validation'


class Preprocessing():
    def setdata(self, train_path, test_path, image_resize, batch_size):
        self.train_path = train_path
        self.test_path = test_path
        self.image_resize = image_resize
        self.batch_size = batch_size

    def image(self):
        train_path, test_path, image_resize, batch_size = self.train_path, self.test_path, self.image_resize, self.batch_size

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            return data, label

        train_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.ImageFolderDataset(train_path, transform = transformer),
            batch_size = batch_size, shuffle = True, last_batch = 'discard')

        test_iter = gluon.data.DataLoader(
            gluon.data.vision.datasets.ImageFolderDataset(test_path, transform = transformer),
            batch_size = batch_size, shuffle = True, last_batch = 'discard')

        return train_iter, test_iter



