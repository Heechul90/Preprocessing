import os
import mxnet as mx
from mxnet import gluon, nd
import numpy as np
from glob import glob


class Preprocessing():
    def setdata(self, data_path, image_resize):
        self.data_path = data_path
        self.image_resize = image_resize

    def load_image(self):
        path, image_resize = self.data_path, self.image_resize

        image_path = os.path.join(path, '*')

        def transform(image):
            resized = mx.image.resize_short(image, image_resize)  # minimum 224x224 images
            cropped, crop_info = mx.image.center_crop(resized, (224, 224))
            normalized = mx.image.color_normalize(cropped.astype(np.float32) / 255,
                                                  mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                  std=mx.nd.array([0.229, 0.224, 0.225]))
            # the network expect batches of the form (N,3,224,224)
            transposed = normalized.transpose((2, 0, 1))  # Transposing from (224, 224, 3) to (3, 224, 224)
            batchified = transposed.expand_dims(axis=0)  # change the shape from (3, 224, 224) to (1, 3, 224, 224)
            return batchified

        image_list = []
        for image in glob(image_path):
            image = mx.image.imread(image)
            image = transform(image)
            image_list.append(image)

        return image_list


