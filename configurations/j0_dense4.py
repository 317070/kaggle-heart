from default import *

import theano.tensor as T
from layers import MuLogSigmaErfLayer, ConvolutionOverAxisLayer
import objectives

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import reshape
from lasagne.layers import DenseLayer
from postprocess import upsample_segmentation
from volume_estimation_layers import GaussianApproximationVolumeLayer
import theano_printer

validate_every = 100
validate_train_set = False
save_every = 100
restart_from_save = False

batches_per_chunk = 1

batch_size = 2
sunny_batch_size = 4
num_chunks_train = 8400

image_size = 64

learning_rate_schedule = {
    0:   0.0003,
    250:  0.00003,
    5000: 0.000003,
    8000: 0.0000003,
}

data_sizes = {
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}

augmentation_params = {
    "rotation": (0, 360),
    "shear": (-10, 10),
    "translation": (-8, 8),
}

def build_model():

    #################
    # Regular model #
    #################
    l0 = InputLayer(data_sizes["sliced:data:ax"])
    l0r = reshape(l0, (-1, 1, ) + data_sizes["sliced:data:ax"][1:])

    # (batch, channel, time, axis, x, y)

    # convolve over time with small filter
    l0t = ConvolutionOverAxisLayer(l0r, num_filters=2, filter_size=(5,), stride=(3,), axis=2, channel=1,
                                   W=lasagne.init.Orthogonal(),
                                   b=lasagne.init.Constant(0.1),
                                   )

    l0r = reshape(l0t, (-1, 1, ) + data_sizes["sliced:data:ax"][-2:])

    # first do the segmentation steps
    l1a = ConvLayer(l0r, num_filters=32, filter_size=(5, 5), stride=2,
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    )
    l1b = ConvLayer(l1a, num_filters=32, filter_size=(5, 5), stride=2,
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    )
    l1b_m = MaxPoolLayer(l1b, pool_size=(2,2))

    l1c = ConvLayer(l1b_m, num_filters=64, filter_size=(3, 3),
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    )
    l1f = ConvLayer(l1c, num_filters=32, filter_size=(3, 3),
                    pad='same',
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))

    l1f_m = MaxPoolLayer(l1f, pool_size=(2,2))

    l1f_r = reshape(l1f_m, (batch_size, 2, 9, 15, 32, 4, 4))


    l0t = ConvolutionOverAxisLayer(l1f_r, num_filters=32, filter_size=(3,), stride=(1,),
                                   axis=3, channel=1,
                                   W=lasagne.init.Orthogonal(),
                                   b=lasagne.init.Constant(0.1),
                                   )

    l_d3 = lasagne.layers.DenseLayer(l0t,
                              num_units=2,
                              nonlinearity=lasagne.nonlinearities.identity)
    l_systole = MuLogSigmaErfLayer(l_d3)

    l_d3b = lasagne.layers.DenseLayer(l0t,
                              num_units=2,
                              nonlinearity=lasagne.nonlinearities.identity)
    l_diastole = MuLogSigmaErfLayer(l_d3b)

    return {
        "inputs":{
            "sliced:data:ax": l0
        },
        "outputs":{
            "systole": l_systole,
            "diastole": l_diastole
        }
    }


def build_objective(l_ins, l_outs):
    return objectives.KaggleObjective(l_outs)

