import mxnet as mx
from mxnet import gluon, autograd, nd
import numpy as np
from sklearn.model_selection import train_test_split


class Preprocessing():
    def setdata(self, data_path, image_resize, test_size):
        self.data_path = data_path
        self.image_resize = image_resize
        self.test_size = test_size

    def image(self):
        data_path, image_resize, test_size = self.data_path, self.image_resize, self.test_size

        def transformer(data, label):
            data = mx.image.imresize(data, image_resize, image_resize)
            data = mx.nd.transpose(data.astype('float32'), (2, 0, 1)) / 255
            label = np.float32(label)
            return data, label

        input_data = gluon.data.vision.datasets.ImageFolderDataset(data_path, transform=transformer)

        img_data = []
        img_label = []
        for d, l in input_data:
            img_data.append(d)
            img_label.append(l)


        train_data, test_data, train_label, test_label = train_test_split(img_data, img_label, test_size=test_size)

        return train_data, test_data, train_label, test_label


