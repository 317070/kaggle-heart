from default import *

import theano.tensor as T
import objectives

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import reshape
from lasagne.layers import DenseLayer
from postprocess import upsample_segmentation

validate_every = 10
save_every = 100
restart_from_save = False

batches_per_chunk = 1

batch_size = 1
chunk_size = batch_size*batches_per_chunk
num_chunks_train = 840

learning_rate_schedule = {
    0:   0.0003,
    10:  0.00003,
    500: 0.000003,
    800: 0.0000003
}


data_sizes = {
    "sliced:data": (batch_size, 30, 15, 100, 100), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (batch_size, 1, 256, 256)
    # TBC with the metadata
}

def build_model():
    l0 = InputLayer(data_sizes["sliced:data"])
    l0r = reshape(l0, (-1, 1, ) + data_sizes["sliced:data"][-2:])

    # first do the segmentation steps
    l1a = ConvLayer(l0r, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l1b = ConvLayer(l1a, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1))
    l1c = ConvLayer(l1b, num_filters=32, filter_size=(3, 3),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    nonlinearity=lasagne.nonlinearities.sigmoid)
    l1d = ConvLayer(l1c, num_filters=1, filter_size=(1, 1),
                    W=lasagne.init.Orthogonal(),
                    b=lasagne.init.Constant(0.1),
                    nonlinearity=lasagne.nonlinearities.sigmoid)

    # then use sum to estimate volume. Sum over last 3 dimensions
    # TODO: this summer should output a distribution over the 600 classes softmaxed!
    # TODO: it should also take into account resolution, etc.
    volume_layer = GaussianApproximationVolumeLayer(reshape(l1d, data_sizes["sliced:data"][:2]+(-1,)),
                                                   pool_size=l1d.output_shape[-1],
                                                   axis=2,
                                                   pool_function=T.sum)

    # then use max and min for systole and diastole
    l_systole = lasagne.layers.FlattenLayer(
                    lasagne.layers.FeaturePoolLayer(volume_layer,
                                                pool_size=volume_layer.output_shape[-1],
                                                axis=1,
                                                pool_function=T.min), outdim=1)

    l_diastole = lasagne.layers.FlattenLayer(
                    lasagne.layers.FeaturePoolLayer(volume_layer,
                                                pool_size=volume_layer.output_shape[-1],
                                                axis=1,
                                                pool_function=T.max), outdim=1)

    return {
        "inputs":{
            "sliced:data": l0,
        },
        "outputs":{
            "systole": l_systole,
            "diastole": l_diastole
        }
    }


def build_objective(l_ins, l_outs):
    return objectives.R2Objective(l_outs)


def postprocess(output):
    output = output.reshape(64, 32, 32)
    return upsample_segmentation(output, (256, 256))
