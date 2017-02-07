import numpy as np
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
MAX_TRAIN_SIZE = 41
MAX_NUM_EPOCHS = 51

# num filters sets for every layers
NUM_FILTERS1_SET = (16, 16, 32)
NUM_FILTERS2_SET = (16, 32, 64)
NUM_FILTERS3_SET = (64, 128, 256)

BATCH_SIZE_SET = (4096, 8192, 12288, 16384)


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

def get_image_blocks(image, gt):
    h, w = gt.shape
    bg_blocks = []
    obj_blocks = []
    for i in range(h):
        for j in range(w):
            im_x, im_y = (i+pad, j+pad)
            block = image[:, im_x-pad:im_x+pad + 1,
            im_y-pad:im_y+pad + 1]
            if gt[i,j] == 0:
                bg_blocks.append(block)
            else:
                obj_blocks.append(block)
    bg_blocks = np.array(bg_blocks).astype(np.float32)
    obj_blocks = np.array(obj_blocks).astype(np.float32)
    print len(bg_blocks) - len(obj_blocks)
    # Align numbers of background samples and object samples
    if len(bg_blocks) - len(obj_blocks) > 0:
        bg_blocks = np.random.choice(bg_blocks, len(obj_blocks),
                                     replace=False)
    else:
        obj_blocks = np.random.choice(obj_blocks, len(bg_blocks),
                                      replace=False)
    print len(bg_blocks) - len(obj_blocks)

    y_bg = np.zeros(len(bg_blocks)).astype(np.int32)
    y_obj = np.ones(len(obj_blocks)).astype(np.int32)

    X = np.hstack((bg_blocks, obj_blocks))
    y = np.hstack((y_bg, y_obj))
    # Shuffle
    permutation = np.random.permutation(len(bg_blocks) + len(obj_blocks))
    return (X[permutation], y[permutation])

# Net for unary terms
class TinyNet:
    # init params
    def __init__(self, PAD, TRAIN_SIZE, NUM_EPOCHS, NUM_FILTERS1,
                 NUM_FILTERS2, NUM_FILTERS3, BATCH_SIZE):
        self.pad = PAD
        self.train_size = TRAIN_SIZE
        self.val_size = 5
        self.num_epochs = NUM_EPOCHS
        self.num_filters1 = NUM_FILTERS1
        self.num_filters2 = NUM_FILTERS2
        self.num_filters3 = NUM_FILTERS3
        self.batch_size = BATCH_SIZE
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
        X_train = np.array([]).astype(np.float32)
        y_train = np.array([]).astype(np.int32)
        X_val = np.array([]).astype(np.float32)
        y_val = np.array([]).astype(np.int32)
        pad = self.pad
        for n in range(len(images)):
            X, y = get_image_blocks(images[n], gts[n])
            if n < self.train_size:
                np.hstack((X_train, X))
                np.hstack((X_train, X))
            elif n >= len(images) - self.val_size:
                np.hstack((X_val, X))
                np.hstack((X_val, X))
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

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
            0: 0.01,
            self.num_epochs//2: 0.001,
            #self.num_epochs-5: 0.0001
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
        for epoch in range(self.num_epochs):
            if epoch in self.lr_schedule:
                lr = np.float32(self.lr_schedule[epoch])
                print(" setting learning rate to %.7f" % lr)
                self.lr.set_value(lr)
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(self.X_train, self.y_train,
                                             self.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(self.X_val, self.y_val,
                                             self.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs,
                                                    time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    def print_params(self):
        print
        print ('='*50)
        print ('PAD = %d' % self.pad)
        print ('INPUT_SHAPE = ', self.input_shape)
        print ('TRAIN_SIZE = %d' % self.train_size)
        print ('NUM_FILTERS1 = %d' % self.num_filters1)
        print ('NUM_FILTERS2 = %d' % self.num_filters2)
        print ('NUM_FILTERS3 = %d' % self.num_filters3)
        print ('NUM_EPOCHS = %d' % self.num_epochs)
        print ('BATCH_SIZE = %d' % self.batch_size)
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
    pad = 5
    images = pad_images(images, pad)
    # From TF to TH order
    images = images.transpose(0,3,1,2)
    train_size = 10 * (pad - 2)
    for batch_size in BATCH_SIZE_SET[:train_size/10 + 1]:
        for ind in range(3):
            num_filters1 = NUM_FILTERS1_SET[ind]
            num_filters2 = NUM_FILTERS2_SET[ind]
            for num_filters3 in NUM_FILTERS3_SET:
                for num_epochs in range(30, MAX_NUM_EPOCHS, 10):
                    # TRAIN
                    model = TinyNet(pad, train_size, num_epochs, num_filters1,
                                    num_filters2, num_filters3, batch_size)
                    model.print_params()
                    model.build_cnn()
                    model.set_data(images, gts)
                    model.set_train_loss()
                    model.set_test_loss()
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

from math import exp


def get_pairwise_term(pix1, pix2, y1, y2):
    A, B, sigma = (1., 1., 1.)
    # Distanse between two pixels
    distanсe = lambda p1, p2: np.mean(p1 - p2)**2
    # Determine whether two marks belong to different objects
    delta = lambda y1, y2: 0 if y1 == y2 else 1
    # Penalty function
    ksi = lambda x, y: A + B * exp(-distanсe(x, y)/(2 * sigma**2))

    return delta(y1, y2) * ksi(pix1, pix2)


def get_unary_terms(model):


def min_cut(image):
    graph = maxflow.Graph[float]()
    graph.add_grid_nodes(image.shape)

def segmentation(unary_model, images):
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
