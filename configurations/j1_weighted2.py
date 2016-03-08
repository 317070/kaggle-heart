from deep_learning_layers import ConvolutionOver2DAxisLayer, MaxPoolOverAxisLayer, MaxPoolOver2DAxisLayer, \
    MaxPoolOver3DAxisLayer, ConvolutionOver3DAxisLayer, ConvolutionOverAxisLayer
from default import *

import theano.tensor as T
from layers import MuLogSigmaErfLayer, CumSumLayer
import objectives

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import reshape
from lasagne.layers import DenseLayer
from postprocess import upsample_segmentation
from volume_estimation_layers import GaussianApproximationVolumeLayer
import theano_printer

validate_every = 10
validate_train_set = False
save_every = 10
restart_from_save = False

batches_per_chunk = 16

batch_size = 1
sunny_batch_size = 4
num_chunks_train = 20000

image_size = 64

learning_rate_schedule = {
    0:     0.01,
    200:     0.001,
}
from preprocess import preprocess_with_augmentation
preprocess = preprocess_with_augmentation

data_sizes = {
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 20 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}

augmentation_params = {
    "rotation": (0, 360),
    "shear": (-1, 1),
    "translation": (-1, 1),
}

def build_model():

    #################
    # Regular model #
    #################
    l0 = InputLayer(data_sizes["sliced:data:ax"])
    l0r = reshape(l0, (-1, 1, ) + data_sizes["sliced:data:ax"][1:])

    # (batch, channel, time, axis, x, y)

    # convolve over time
    l1 = ConvolutionOverAxisLayer(l0r, num_filters=4, filter_size=(3,), axis=(2,), channel=1,
                                   W=lasagne.init.Orthogonal("relu"),
                                   b=lasagne.init.Constant(0.1),
                                   )
    l1b = ConvolutionOverAxisLayer(l1, num_filters=4, filter_size=(3,), axis=(2,), channel=1,
                                   W=lasagne.init.Orthogonal("relu"),
                                   b=lasagne.init.Constant(0.1),
                                   )
    l1m = MaxPoolOverAxisLayer(l1b, pool_size=(2,), axis=(2,))

    # convolve over x and y
    l2a = ConvolutionOver2DAxisLayer(l1m, num_filters=32, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal("relu"),
                                     b=lasagne.init.Constant(0.1),
                                     )
    l2b = ConvolutionOver2DAxisLayer(l2a, num_filters=32, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal("relu"),
                                     b=lasagne.init.Constant(0.1),
                                     )
    l2m = MaxPoolOver2DAxisLayer(l2b, pool_size=(2, 2), axis=(4,5))

    # convolve over x, y and axis
    l3a = ConvolutionOver3DAxisLayer(l2m, num_filters=64, filter_size=(3, 3, 3),
                                     axis=(3,4,5), channel=1,
                                     W=lasagne.init.Orthogonal("relu"),
                                     b=lasagne.init.Constant(0.1),
                                     )

    l3b = ConvolutionOver3DAxisLayer(l3a, num_filters=64, filter_size=(3, 3, 3),
                                     axis=(3,4,5), channel=1,
                                     W=lasagne.init.Orthogonal("relu"),
                                     b=lasagne.init.Constant(0.1),
                                     )
    l3c = ConvolutionOver3DAxisLayer(l3b, num_filters=64, filter_size=(3, 3, 3),
                                     axis=(3,4,5), channel=1,
                                     W=lasagne.init.Orthogonal("relu"),
                                     b=lasagne.init.Constant(0.1),
                                     )
    l3m = MaxPoolOver3DAxisLayer(l3c, pool_size=(2, 2, 2), axis=(3,4,5))

    # convolve over time
    l4 = ConvolutionOverAxisLayer(l3m, num_filters=64, filter_size=(3,), axis=(2,), channel=1,
                                   W=lasagne.init.Orthogonal("relu"),
                                   b=lasagne.init.Constant(0.1),
                                   )
    l4m = MaxPoolOverAxisLayer(l4, pool_size=(2,), axis=(2,))

    # convolve over axis
    l5 = ConvolutionOverAxisLayer(l4m, num_filters=128, filter_size=(3,), axis=(3,), channel=1,
                                   W=lasagne.init.Orthogonal("relu"),
                                   b=lasagne.init.Constant(0.1),
                                   )

    # convolve over x and y
    l6a = ConvolutionOver2DAxisLayer(l5, num_filters=128, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal("relu"),
                                     b=lasagne.init.Constant(0.1),
                                     )
    l6b = ConvolutionOver2DAxisLayer(l6a, num_filters=128, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal("relu"),
                                     b=lasagne.init.Constant(0.1),
                                     )
    l6m = MaxPoolOver2DAxisLayer(l6b, pool_size=(2, 2), axis=(4,5))

    # convolve over time and x,y
    l7 = ConvolutionOver3DAxisLayer(l6m, num_filters=32, filter_size=(3,3,3), axis=(2,4,5), channel=1,
                                   W=lasagne.init.Orthogonal("relu"),
                                   b=lasagne.init.Constant(0.1),
                                   )

    l8 = lasagne.layers.DropoutLayer(l7)

    l_d3a = lasagne.layers.DenseLayer(l8,
                              num_units=600)

    l_d3b = lasagne.layers.DropoutLayer(l_d3a)

    l_systole = lasagne.layers.DenseLayer(l_d3b,
                              num_units=600,
                              nonlinearity=lasagne.nonlinearities.softmax)


    l_d3c = lasagne.layers.DenseLayer(l8,
                              num_units=600)

    l_d3d = lasagne.layers.DropoutLayer(l_d3c)

    l_diastole = lasagne.layers.DenseLayer(l_d3d,
                              num_units=600,
                              nonlinearity=lasagne.nonlinearities.softmax)

    return {
        "inputs":{
            "sliced:data:ax": l0
        },
        "outputs": {
            "systole:onehot": l_systole,
            "diastole:onehot": l_diastole,
        },
        "regularizable": {
            l_d3a: 0.25,
            l_d3c: 0.25,
            l_systole: 0.25,
            l_diastole: 0.25,
        }
    }


def build_objective(interface_layers):
    l2_penalty = 0#lasagne.regularization.regularize_layer_params_weighted(interface_layers["regularizable"], lasagne.regularization.l2)
    return objectives.WeightedLogLossObjective(interface_layers["outputs"], penalty=l2_penalty)
