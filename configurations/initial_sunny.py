from default import *

import theano.tensor as T
import objectives

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from postprocess import upsample_segmentation

validate_every = 10
save_every = 100
restart_from_save = False

batches_per_chunk = 64

batch_size = 64
chunk_size = batch_size*batches_per_chunk

num_chunks_train = 840

learning_rate_schedule = {
    0:   0.0003,
    10:  0.00003,
    500: 0.000003,
    800: 0.0000003
}


data_sizes = {
    "sliced:data": (batch_size, 30, 30, 100, 100), #30 mri_slices, 30 time steps, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (batch_size, 1, 256, 256)
    # TBC with the metadata
}

def build_model():
    l0 = InputLayer(data_sizes["sunny"])
    l1a = ConvLayer(l0, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l1b = ConvLayer(l1a, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l1c = ConvLayer(l1b, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l1 = MaxPoolLayer(l1c, pool_size=(2, 2))

    l2a = ConvLayer(l1, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l2b = ConvLayer(l2a, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l2c = ConvLayer(l2b, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l3 = MaxPoolLayer(l2c, pool_size=(3, 3))

    l4a = ConvLayer(l3, num_filters=32, filter_size=(4, 4),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l4b = ConvLayer(l4a, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l4c = ConvLayer(l4b, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))

    l5a = ConvLayer(l4c, num_filters=256, filter_size=(1, 1),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l5b = ConvLayer(l5a, num_filters=256, filter_size=(1, 1),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l5c = ConvLayer(l5b, num_filters=1, filter_size=(1, 1),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    nonlinearity=lasagne.nonlinearities.sigmoid)

    l_final = lasagne.layers.FlattenLayer(l5c, outdim=2)

    return {
        "inputs":{
            "sunny": l0
        },
        "outputs":{
            "segmentation": l_final,
            "top": l_final
        }
    }


def build_objective(l_ins, l_outs):
    return objectives.UpscaledImageObjective(l_outs)


def postprocess(output):
    output = output.reshape(64, 32, 32)
    return upsample_segmentation(output, (256, 256))
