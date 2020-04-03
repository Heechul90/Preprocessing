import json
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np


########################################################################################################################
from cnn.load_image import Preprocessing

path = 'dataset/image/pretrained/'
image_resize = 224

a = Preprocessing()
a.setdata(path, image_resize)
image_list = a.load_image()
len(image_list)


########################### Loading the model
ctx = mx.cpu()

from mxnet.gluon.model_zoo import vision
resnet = vision.resnet18_v1(pretrained=True, ctx=ctx)
alexnet = vision.alexnet(pretrained=True, ctx=ctx)


### Loading the data
categories = np.array(json.load(open('image_net_labels.json', 'r')))

def predict(model, image, categories, k):
    predictions = model(image).softmax()
    top_pred = predictions.topk(k=k)[0].asnumpy()
    for index in top_pred:
        probability = predictions[0][int(index)]
        category = categories[int(index)]
        print("{}: {:.2f}%".format(category, probability.asscalar()*100))
    print('')


for image in image_list:
    predict(resnet, image, categories, 3)








