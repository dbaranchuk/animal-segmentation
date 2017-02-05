import numpy as np
import keras
import maxflow
from skimage.util import pad

from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, ElemwiseSumLayer, MaxPool2DLayer
from lasagne.init import HeNormal
from lasagne.nonlinearities import rectify, softmax

INPUT_SHAPE = (None, 3, 11, 11)
PAD = 5
TRAIN_SIZE = 40
VAL_SIZE = 10

def residual_block(layer, num_filters):
    #Bottleneck
    l_conv1 = Conv2DLayer(layer, num_filters/4, filter_size=1, pad='same',
                        nonlinearity=rectify, W=HeNormal())
    l_conv2 = Conv2DLayer(l_conv1, num_filters/4, filter_size=3, pad='same',
                        nonlinearity=rectify, W=HeNormal())
    l_conv3 = Conv2DLayer(l_conv2, num_filters, filter_size=1, pad='same',
                        nonlinearity=rectify, W=HeNormal())
    return ElemwiseSumLayer([l_conv3, layer])



class TinyResNet:
    def __init__(self):
        l_in = InputLayer(INPUT_SHAPE)
        l_conv1 = Conv2DLayer(l_in, num_filters=32, filter_size=3,
                            nonlinearity=rectify, W=HeNormal())
        l_conv2 = Conv2DLayer(l_conv1, num_filters=64, filter_size=2, stride=2,
                            nonlinearity=rectify, W=HeNormal())
        l_res = residual_block(l_conv2, 64)
        l_max = MaxPool2DLayer(l_res, pool_size=(2, 2))
        l_dense = DenseLayer(l_max, num_units=512, nonlinearity=rectify)
        l_out = DenseLayer(l_dense, num_units=2, nonlinearity=softmax)
        self.model = l_out

    def set_data(self, images, gts):
        X_data = []
        y_data = []
        for n in range(len(images)):
            image, gt = (images[n], gts[n])
            h, w = gts[n].shape
            for i in range(h):
                for j in range(w):
                    im_x, im_y = (i+PAD, j+PAD)
                    block = image[:, im_x-PAD:im_x+PAD+1, im_y-PAD:im_y+PAD+1]
                    X_data.append(block)
                    y_data.append(gt[i,j])
        X_data, y_data = (np.array(X_data), np.array(y_data))
        self.X_train = X_data[:TRAIN_SIZE]
        self.y_train = y_data[:TRAIN_SIZE]
        self.X_val = X_data[TRAIN_SIZE:VAL_SIZE+TRAIN_SIZE]
        self.y_val = y_data[TRAIN_SIZE:VAL_SIZE+TRAIN_SIZE]

        print(self.X_train.shape, self.y_train.shape, self.X_val.shape, self.y_val.shape)


def pad_images(images):
    new_images = []
    for image in images:
        h, w, c = image.shape
        new_image = np.zeros((h+2*PAD, w+2*PAD, c))
        for i in range(image.shape[2]):
            new_image[...,i] = np.lib.pad(image[...,i], (5,5), 'reflect')
        new_images.append(new_image)
    return np.array(new_images)

def train_unary_model(images, gts):
    print(gts[0].shape)
    images = pad_images(images)
    # From TF to TH order
    images = images.transpose(0,3,1,2)

    model = TinyResNet()
    model.set_data(images, gts)
    return {}

def segmentation(unary_model, images):
    return [np.zeros(img.shape[:2]) for img in images]
