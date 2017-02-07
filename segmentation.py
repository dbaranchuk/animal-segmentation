#-*- coding: utf-8 -*-
import numpy as np
from math import exp
import maxflow
import time
import theano
import theano.tensor as T
from skimage.util import pad

from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, \
                          ElemwiseSumLayer, MaxPool2DLayer,     \
                          get_output, get_all_params, batch_norm
from lasagne.init import HeNormal
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import nesterov_momentum


# Worker
MAX_NUM_EPOCHS = 61

# num filters sets for every layers
NUM_FILTERS1_SET = (16, 16)
NUM_FILTERS2_SET = (16, 32)
NUM_FILTERS3_SET = (128, 256)

PAD = 5
BATCH_SIZE = 16384 #(8192, 12288, 16384)
TRAIN_SIZE = 45
VAL_SIZE = 5
NUM_EPOCHS = 50

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


def get_image_blocks(image, gt=None):
    h, w = gt.shape
    blocks = []
    bg_blocks = []
    obj_blocks = []
    for i in range(h):
        for j in range(w):
            im_x, im_y = (i+PAD, j+PAD)
            block = image[:, im_x-PAD:im_x+PAD + 1,
                          im_y-PAD:im_y+PAD + 1]
            if gt is None:
                blocks.append(block)
            elif gt[i,j] == 0:
                bg_blocks.append(block)
            else:
                obj_blocks.append(block)
    if gt is None:
        return np.array(blocks).astype(np.float32)
    else:
        return (np.array(bg_blocks).astype(np.float32),
               np.array(obj_blocks).astype(np.float32))


def get_data(image, gt):
    bg_blocks, obj_blocks = get_image_blocks(image, gt)
    # Align numbers of background samples and object samples
    if len(bg_blocks) - len(obj_blocks) > 0:
        inds = np.arange(len(bg_blocks))
        inds = np.random.choice(inds, len(obj_blocks), replace=False)
        bg_blocks = bg_blocks[inds]
    else:
        inds = np.arange(len(obj_blocks))
        inds = np.random.choice(inds, len(bg_blocks), replace=False)
        obj_blocks = obj_blocks[inds]
    y_bg = np.zeros(len(bg_blocks)).astype(np.int32)
    y_obj = np.ones(len(obj_blocks)).astype(np.int32)
    X = np.concatenate((bg_blocks, obj_blocks), axis=0)
    y = np.concatenate((y_bg, y_obj), axis=0)
    # Shuffle
    permutation = np.random.permutation(2*len(bg_blocks))
    return (X[permutation], y[permutation])


# Net for unary terms
class TinyNet:
    def __init__(self):
        return
    # init params
    def set_params(self, NUM_FILTERS1, NUM_FILTERS2, NUM_FILTERS3):
        self.num_filters1 = NUM_FILTERS1
        self.num_filters2 = NUM_FILTERS2
        self.num_filters3 = NUM_FILTERS3
        self.input_shape = (None, 3, 2*PAD + 1, 2*PAD + 1)
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
    
    # Build Net
    def build_cnn(self):
        # Input Layer
        l_in = InputLayer(self.input_shape, input_var=self.input_var)
        # Conv1
        l_conv1 = Conv2DLayer(l_in, num_filters=self.num_filters1, filter_size=3,
                              nonlinearity=rectify, W=HeNormal())
        l_conv1 = batch_norm(l_conv1)
        # Conv2
        l_conv2 = Conv2DLayer(l_conv1, num_filters=self.num_filters2, filter_size=2,
                              stride=2, nonlinearity=rectify, W=HeNormal())
        l_conv2 = batch_norm(l_conv2)
        # Pool
        l_max = MaxPool2DLayer(l_conv2, pool_size=(2, 2))
        # FC
        l_dense = DenseLayer(l_max, num_units=self.num_filters3, nonlinearity=rectify)
        # Softmax Output
        l_out = DenseLayer(l_dense, num_units=2, nonlinearity=softmax)
        self.model = l_out

    #Produce train/val data from input images
    def set_data(self, images, gts):
        X_train, y_train = ([], [])
        X_val, y_val = ([], [])
        for n in range(len(images)):
            X, y = get_data(images[n], gts[n])
            if n < TRAIN_SIZE:
                X_train += list(X)
                y_train += list(y)
            elif n >= len(images) - VAL_SIZE:
                X_val += list(X)
                y_val += list(y)
        self.X_train = np.array(X_train).astype(np.float32)
        self.y_train = np.array(y_train).astype(np.int32)
        self.X_val = np.array(X_val).astype(np.float32)
        self.y_val = np.array(y_val).astype(np.int32)
        print(self.X_train.shape, self.y_train.shape,
              self.X_val.shape, self.y_val.shape)

    # Set a loss expression for training
    def set_train_loss(self):
        prediction = get_output(self.model)
        loss = categorical_crossentropy(prediction, self.target_var)
        self.train_loss = loss.mean()

    # Set a loss expression for validation/testing (ignoring dropout during the forward pass)
    def set_val_loss(self):
        prediction = get_output(self.model, deterministic=True)
        loss = categorical_crossentropy(prediction, self.target_var)
        self.val_loss = loss.mean()
        self.val_acc = T.mean(T.eq(T.argmax(prediction, axis=1), self.target_var),
                               dtype=theano.config.floatX)

    # Set learning rate and Nesterov Momentum as update method
    def set_update(self):
        params = get_all_params(self.model, trainable=True)
        self.lr_schedule = {
            0: 0.01,
            20: 0.001,
            #NUM_EPOCHS-5: 0.0001
        }
        self.lr = theano.shared(np.float32(self.lr_schedule[0]))
        self.updates = nesterov_momentum(self.train_loss, params,
                                         learning_rate=self.lr, momentum=0.9)

    # Training / validation process
    def train(self):
        train_fn = theano.function([self.input_var, self.target_var],
                                   self.train_loss, updates=self.updates)
        val_fn = theano.function([self.input_var, self.target_var],
                                 [self.val_loss, self.val_acc])
        for epoch in range(NUM_EPOCHS):
            if epoch in self.lr_schedule:
                lr = np.float32(self.lr_schedule[epoch])
                print(" setting learning rate to %.7f" % lr)
                self.lr.set_value(lr)
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(self.X_train, self.y_train,
                                             BATCH_SIZE, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(self.X_val, self.y_val,
                                             BATCH_SIZE//8, shuffle=False):
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

    def get_predictions(self, image):
        prediction = get_output(self.model, deterministic=True)
        test_fn = theano.function([self.input_var], prediction)
        blocks = get_image_blocks(image)

        preds = []
        for block in blocks:
            preds.append(test_fn(block))
        print preds

    def print_params(self):
        print
        print ('='*50)
        print ('PAD = %d' % PAD)
        print ('INPUT_SHAPE = ', self.input_shape)
        print ('NUM_FILTERS1 = %d' % self.num_filters1)
        print ('NUM_FILTERS2 = %d' % self.num_filters2)
        print ('NUM_FILTERS3 = %d' % self.num_filters3)
        print ('NUM_EPOCHS = %d' % NUM_EPOCHS)
        print ('='*50)
        print

# Extend borders for extriving blocks for every pixel
def pad_images(images, pad):
    new_images = []
    for image in images:
        h, w, c = image.shape
        new_image = np.zeros((h + 2*pad, w + 2*pad, c))
        for i in range(image.shape[2]):
            new_image[...,i] = np.lib.pad(image[...,i], (pad, pad), 'reflect')
        new_images.append(new_image)
    return np.array(new_images)

# WORKER
def train_unary_model(images, gts):
    images = pad_images(images, PAD)
    images = images.transpose(0,3,1,2)
    model = TinyNet()
    model.set_data(images, gts)

    for ind in range(2):
        num_filters1 = NUM_FILTERS1_SET[ind]
        num_filters2 = NUM_FILTERS2_SET[ind]
        for num_filters3 in NUM_FILTERS3_SET:
            # TRAIN
            model.set_params(num_filters1, num_filters2, num_filters3)
            model.print_params()
            model.build_cnn()
            model.set_train_loss()
            model.set_val_loss()
            model.set_update()
            model.train()
    return {}

# Main training function
#def train_unary_model(images, gts):
#    images = pad_images(images)
    # From TF to TH order
#    images = images.transpose(0,3,1,2)

#    model = TinyNet()
#    model.set_data(images, gts)
#    model.set_train_loss()
#    model.set_test_loss()
#    model.set_update()
#    model.train()
#    return {}

def compute_weights(X, Y):
    A, B, sigma = (1., 1., 1.)
    distance = lambda x, y: np.mean((x - y)**2, axis=0)
    penalty_fn = lambda x: A + B * np.exp(-x / (2 * sigma**2))
    dist = distance(X, Y)
    return penalty_fn(dist)


def min_cut(image):
    graph = maxflow.Graph[float]()
    nodeids = graph.add_grid_nodes(image.shape)

    # Set Unary Terms

    # Set Pairwise Terms
    # Compute Horizontal Weights
    h, w = image.shape
    h_image_1 = image.copy()[1:, :]
    h_image_2 = image.copy()[:-1, :]
    h_weights = compute_weights(h_image_1, h_image_2)
    h_struct = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 1, 0]])
    graph.add_grid_edges(nodeids, h_weights, h_struct)

    # Compute Vertical Weights
    v_image_1 = image.copy()[:, 1:]
    v_image_2 = image.copy()[:, :-1]
    v_weights = compute_weights(v_image_1, v_image_2)
    v_struct = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 0, 0]])
    graph.add_grid_edges(nodeids, v_weights, v_struct)

    graph.maxflow()
    sgm = graph.get_grid_segments(nodeids)


def segmentation(unary_model, images):
    images = pad_images(images, 5)
    # From TF to TH order
    images = images.transpose(0,3,1,2)

    return [np.zeros(img.shape[:2]) for img in images]













'''
    # Bottleneck
    def residual_block(layer, num_filters):
    l_conv1 = Conv2DLayer(layer, num_filters/4, filter_size=1, pad='same',
    nonlinearity=rectify, W=HeNormal())
    l_conv1 = batch_norm(l_conv1)

    l_conv2 = Conv2DLayer(l_conv1, num_filters/4, filter_size=3, pad='same',
    nonlinearity=rectify, W=HeNormal())
    l_conv2 = batch_norm(l_conv2)

    l_conv3 = Conv2DLayer(l_conv2, num_filters, filter_size=1, pad='same',
    nonlinearity=rectify, W=HeNormal())
    l_conv3 = batch_norm(l_conv3)
    return ElemwiseSumLayer([l_conv3, layer])
'''
