import numpy as np
import keras
import maxflow
import time
import theano
import theano.tensor as T
from skimage.util import pad

from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, \
                          ElemwiseSumLayer, MaxPool2DLayer,    \
                          get_output, get_all_params
from lasagne.init import HeNormal
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import nesterov_momentum

INPUT_SHAPE = (None, 3, 11, 11)
PAD = 5
TRAIN_SIZE = 15 #40
VAL_SIZE = 5 #10
NUM_EPOCHS = 15


# Bottleneck
def residual_block(layer, num_filters):
    l_conv1 = Conv2DLayer(layer, num_filters/4, filter_size=1, pad='same',
                        nonlinearity=rectify, W=HeNormal())
    l_conv2 = Conv2DLayer(l_conv1, num_filters/4, filter_size=3, pad='same',
                        nonlinearity=rectify, W=HeNormal())
    l_conv3 = Conv2DLayer(l_conv2, num_filters, filter_size=1, pad='same',
                        nonlinearity=rectify, W=HeNormal())
    return ElemwiseSumLayer([l_conv3, layer])


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# Net for unary terms
class TinyResNet:
    # Build Net
    def __init__(self):
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')

        l_in = InputLayer(INPUT_SHAPE, input_var=self.input_var)
        l_conv1 = Conv2DLayer(l_in, num_filters=32, filter_size=3,
                            nonlinearity=rectify, W=HeNormal())
        l_conv2 = Conv2DLayer(l_conv1, num_filters=64, filter_size=2, stride=2,
                            nonlinearity=rectify, W=HeNormal())
        l_res = residual_block(l_conv2, 64)
        l_max = MaxPool2DLayer(l_res, pool_size=(2, 2))
        l_dense = DenseLayer(l_max, num_units=512, nonlinearity=rectify)
        l_out = DenseLayer(l_dense, num_units=2, nonlinearity=softmax)
        self.model = l_out

    #Produce train/val data from input images
    def set_data(self, images, gts):
        self.X_train, self.y_train = ([],[])
        self.X_val, self.y_val = ([],[])
        for n in range(len(images)):
            image, gt = (images[n], gts[n])
            h, w = gts[n].shape
            for i in range(h):
                for j in range(w):
                    im_x, im_y = (i+PAD, j+PAD)
                    block = image[:, im_x-PAD:im_x+PAD+1, im_y-PAD:im_y+PAD+1]
                    if n < TRAIN_SIZE:
                        self.X_train.append(block)
                        self.y_train.append(gt[i,j])
                    else:
                        self.X_val.append(block)
                        self.y_val.append(gt[i,j])
        self.X_train = np.array(self.X_train).astype(np.float32)
        self.y_train = np.array(self.y_train).astype(np.int32)
        self.X_val = np.array(self.X_val).astype(np.float32)
        self.y_val = np.array(self.y_val).astype(np.int32)
        print(len(self.X_train), len(self.y_train), len(self.X_val), len(self.y_val))

    # Set a loss expression for training
    def set_train_loss(self):
        prediction = get_output(self.model)
        loss = categorical_crossentropy(prediction, self.target_var)
        self.train_loss = loss.mean()

    # Set a loss expression for validation/testing (ignoring dropout during the forward pass)
    def set_test_loss(self):
        prediction = get_output(self.model, deterministic=True)
        loss = categorical_crossentropy(prediction, self.target_var)
        self.test_loss = loss.mean()
        self.test_acc = T.mean(T.eq(T.argmax(prediction, axis=1), self.target_var),
                               dtype=theano.config.floatX)

    # Set learning rate and Nesterov Momentum as update method
    def set_update(self):
        params = get_all_params(self.model, trainable=True)
        self.lr_schedule = {
            0: 0.0001,
            10: 0.00001
        }
        self.lr = theano.shared(np.float32(self.lr_schedule[0]))
        self.updates = nesterov_momentum(self.train_loss, params,
                                         learning_rate=self.lr, momentum=0.9)

    # Training / validation process
    def train(self):
        train_fn = theano.function([self.input_var, self.target_var],
                                   self.train_loss, updates=self.updates)
        val_fn = theano.function([self.input_var, self.target_var],
                                 [self.test_loss, self.test_acc])
        for epoch in range(NUM_EPOCHS):
            if epoch in self.lr_schedule:
                lr = np.float32(self.lr_schedule[epoch])
                print(" setting learning rate to %.7f" % lr)
                self.lr.set_value(lr)
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(self.X_train, self.y_train,
                                             512, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(self.X_val, self.y_val,
                                             512, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, NUM_EPOCHS,
                                                    time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))


# Extend borders for extriving blocks for every pixel
def pad_images(images):
    new_images = []
    for image in images:
        h, w, c = image.shape
        new_image = np.zeros((h+2*PAD, w+2*PAD, c))
        for i in range(image.shape[2]):
            new_image[...,i] = np.lib.pad(image[...,i], (5,5), 'reflect')
        new_images.append(new_image)
    return np.array(new_images)


# Main training function
def train_unary_model(images, gts):
    print(gts[0].shape)
    images = pad_images(images)
    # From TF to TH order
    images = images.transpose(0,3,1,2)

    model = TinyResNet()
    model.set_data(images, gts)
    model.set_train_loss()
    model.set_test_loss()
    model.set_update()
    model.train()
    return {}


def segmentation(unary_model, images):
    return [np.zeros(img.shape[:2]) for img in images]
