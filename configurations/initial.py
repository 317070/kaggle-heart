#TODO: add code
from default import *

import theano.tensor as T
import objectives

from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer

def build_model():
    l0 = InputLayer((batch_size, 1, 255, 255))
    l1a = ConvLayer(l0, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l1b = ConvLayer(l1a, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l1 = MaxPoolLayer(l1b, pool_size=(2, 2))

    l2a = ConvLayer(l1, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l2b = ConvLayer(l2a, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l2 = MaxPoolLayer(l2b, pool_size=(2, 2))

    l3a = ConvLayer(l2, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l3b = ConvLayer(l3a, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l3 = MaxPoolLayer(l3b, pool_size=(2, 2))

    l4a = ConvLayer(l3, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l4b = ConvLayer(l4a, num_filters=32, filter_size=(3, 3),
                                               b=lasagne.init.Constant(0.1))
    l4 = MaxPoolLayer(l4b, pool_size=(2, 2))

    l5 = DenseLayer(l4, num_units=64)

    l6 = DenseLayer(lasagne.layers.dropout(l5, p=0.5), num_units=128)

    l7 = DenseLayer(lasagne.layers.dropout(l6, p=0.5), num_units=32*32, nonlinearity=T.nnet.sigmoid)

    return {
        "inputs":[l0],
        "output":l7
    }


def build_objective(l_ins, l_out):
    return objectives.UpscaledImageObjective(l_out)