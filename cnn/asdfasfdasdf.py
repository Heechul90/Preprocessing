def transform(image):
    resized = mx.image.resize_short(image, 224) #minimum 224x224 images  # shape of inceptionv2 should be 299x299
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped.astype(np.float32)/255,
                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
                                      std=mx.nd.array([0.229, 0.224, 0.225]))
    # the network expect batches of the form (N,3,224,224)
    transposed = normalized.transpose((2,0,1))  # Transposing from (224, 224, 3) to (3, 224, 224)
    batchified = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified


# Testing the different networks
# NUM_CLASSES = 10
# with net.name_scope():
#     net.output = gluon.nn.Dense(NUM_CLASSES)

#filename = mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/onnx/images/dog.jpg?raw=true', fname='dog.jpg')
#filename = 'dog.jpg'
#pathInfo = '../cifar10/'
pathInfo = '../imagenet/'
FileList = [pathInfo+file for file in os.listdir(pathInfo)]

i = 0
image = mx.image.imread('../cifar10/277_cat.png')
imagefiles = []
for file in FileList:
    if i > 10:
        break
    i = i+1
    #img=mx.image.imread(file)
    imagefiles.append(file)

for file in imagefiles :
    img=mx.image.imread(file)
    img = transform_eval(img)
# print(net.output)