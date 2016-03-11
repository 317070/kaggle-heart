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
from updates import build_adam_updates

validate_every = 10
validate_train_set = False
save_every = 10
restart_from_save = False

create_test_gen = partial(generate_test_batch, set="test")  # validate as well by default

dump_network_loaded_data = False

batches_per_chunk = 1

batch_size = 128
sunny_batch_size = 4
num_epochs_train = 2000

image_size = 64

learning_rate_schedule = {
    0:     0.0001,
}

from preprocess import preprocess, preprocess_with_augmentation
from postprocess import postprocess_onehot
preprocess_train = preprocess_with_augmentation
preprocess_validation = preprocess  # no augmentation
preprocess_test = preprocess_with_augmentation
test_time_augmentations = 100

build_updates = build_adam_updates
postprocess = postprocess

data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}

def build_model():

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:singleslice:difference"]

    l0 = InputLayer(input_size)
    # add channel layer
    # l0r = reshape(l0, (-1, 1, ) + input_size[1:])

    # (batch, channel, time, x, y)

    l = ConvolutionOver2DAxisLayer(l0, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )

    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )

    l = BatchNormLayer(l, gamma=None)
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    l = MaxPoolOver2DAxisLayer(l, pool_size=(2, 2), axis=(2,3), stride=(2,2))

    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = BatchNormLayer(l, gamma=None)
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    l = MaxPoolOver2DAxisLayer(l, pool_size=(2, 2), axis=(2,3), stride=(2,2))

    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = BatchNormLayer(l, gamma=None)
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    l = MaxPoolOver2DAxisLayer(l, pool_size=(2, 2), axis=(2,3), stride=(2,2))


    l_dense = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(l),
                              num_units=600,
                              nonlinearity=lasagne.nonlinearities.softmax)

    l_systole = CumSumLayer(l_dense)

    #===================================================================================


    l = ConvolutionOver2DAxisLayer(l0, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )

    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )

    l = BatchNormLayer(l, gamma=None)
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    l = MaxPoolOver2DAxisLayer(l, pool_size=(2, 2), axis=(2,3), stride=(2,2))

    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = BatchNormLayer(l, gamma=None)
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    l = MaxPoolOver2DAxisLayer(l, pool_size=(2, 2), axis=(2,3), stride=(2,2))

    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = ConvolutionOver2DAxisLayer(l, num_filters=64, filter_size=(3, 3),
                                     axis=(2,3), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     nonlinearity=lasagne.nonlinearities.identity
                                     )
    l = BatchNormLayer(l, gamma=None)
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    l = MaxPoolOver2DAxisLayer(l, pool_size=(2, 2), axis=(2,3), stride=(2,2))


    l_dense = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(l),
                              num_units=600,
                              nonlinearity=lasagne.nonlinearities.softmax)

    l_diastole = CumSumLayer(l_dense)

    return {
        "inputs":{
            "sliced:data:singleslice:difference": l0
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        }
    }


def build_objective(interface_layers):
    return objectives.KaggleObjective(interface_layers["outputs"])

