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
from lasagne.layers import BatchNormLayer
from postprocess import upsample_segmentation
from volume_estimation_layers import GaussianApproximationVolumeLayer
import theano_printer
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from updates import build_adam_updates

validate_every = 100
validate_train_set = False
save_every = 100
restart_from_save = False

dump_network_loaded_data = False

batches_per_chunk = 32

batch_size = 32
sunny_batch_size = 4
num_chunks_train = 20000

image_size = 64

learning_rate_schedule = {
    0:    0.0001,
}

from preprocess import preprocess, preprocess_with_augmentation
from postprocess import postprocess_onehot
preprocess_train = preprocess_with_augmentation  # no augmentation
preprocess_validation = preprocess  # no augmentation
preprocess_test = preprocess  # no augmentation

build_updates = build_adam_updates

data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}

augmentation_params = {
    "rotation": (-15, 15),
    "shear": (0, 0),
    "translation": (0, 0),
}

class NormalizationLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return (input - T.mean(input, axis=[-2, -1], keepdims=True)) / (T.std(input, axis=[-2, -1], keepdims=True) + 1e-7)


def build_model():

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:singleslice"]

    l0 = InputLayer(input_size)
    # add channel layer
    #l0r = reshape(l0, (-1, 1, ) + input_size[1:])

    # (batch, channel, time, x, y)
    l = Conv2DDNNLayer(l0, num_filters=64, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal('relu'),
                       b=lasagne.init.Constant(0.1),
                       pad='same')
    l = Conv2DDNNLayer(l, num_filters=64, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1),
                       pad="valid")

    l = lasagne.layers.PadLayer(l, width=(1, 1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    #l = lasagne.layers.DropoutLayer(l, p=0.25)

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1),
                       pad="same")
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1),
                       pad="valid")

    l = lasagne.layers.PadLayer(l, width=(1, 1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    #l = lasagne.layers.DropoutLayer(l, p=0.25)

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(2, 2),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1))
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(2, 2),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    l = lasagne.layers.DropoutLayer(l, p=0.25)

    # --------------------------------------------------------------

    l = lasagne.layers.FlattenLayer(l)
    l_d1 = lasagne.layers.DenseLayer(l, num_units=1024, W=lasagne.init.Orthogonal('relu'), b=lasagne.init.Constant(0.1))
    l_systole = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_d1, p=0.5), num_units=1, W=lasagne.init.Orthogonal('relu'),
                                      b=lasagne.init.Constant(0.1), nonlinearity=lasagne.nonlinearities.identity)

    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # --------------------------------------------------------------


    l = Conv2DDNNLayer(l0, num_filters=64, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal('relu'),
                       b=lasagne.init.Constant(0.1),
                       pad='same')
    l = Conv2DDNNLayer(l, num_filters=64, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1),
                       pad="valid")

    l = lasagne.layers.PadLayer(l, width=(1, 1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    #l = lasagne.layers.DropoutLayer(l, p=0.25)

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1),
                       pad="same")
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1),
                       pad="valid")

    l = lasagne.layers.PadLayer(l, width=(1, 1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    #l = lasagne.layers.DropoutLayer(l, p=0.25)

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(2, 2),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1))
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(2, 2),
                       W=lasagne.init.Orthogonal("relu"),
                       b=lasagne.init.Constant(0.1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    l = lasagne.layers.DropoutLayer(l, p=0.25)

    # --------------------------------------------------------------

    l = lasagne.layers.FlattenLayer(l)
    l_d2 = lasagne.layers.DenseLayer(l, num_units=1024, W=lasagne.init.Orthogonal('relu'), b=lasagne.init.Constant(0.1))
    l_diastole = lasagne.layers.DenseLayer(lasagne.layers.dropout(l_d2, p=0.5), num_units=1, W=lasagne.init.Orthogonal('relu'),
                                      b=lasagne.init.Constant(0.1), nonlinearity=lasagne.nonlinearities.identity)

    return {
        "inputs":{
            "sliced:data:singleslice": l0
        },
        "outputs": {
            "systole:value": l_systole,
            "diastole:value": l_diastole,
        },
        "regularizable": {
            l_d1: 1e-3,
            l_d2: 1e-3,
        }
    }


def build_objective(interface_layers):
    l2_penalty = lasagne.regularization.regularize_layer_params_weighted(interface_layers["regularizable"], lasagne.regularization.l2)

    return objectives.KaggleValidationMSEObjective(interface_layers["outputs"], penalty=l2_penalty)

