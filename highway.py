import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
from deep_learning_layers import *

class MultiplicativeGatingLayer(nn.layers.MergeLayer):
    """
    Generic layer that combines its 3 inputs t, h1, h2 as follows:
    y = t * h1 + (1 - t) * h2
    """
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)

        self.smallest_shape =  tuple([min(a,b,c) for a,b,c in zip(gate.output_shape, input1.output_shape, input2.output_shape)])
        self.slices = []
        for input in incomings:
            input_slicing = []
            for dim in xrange(len(input.output_shape)):
                diff = input.output_shape[dim] - self.smallest_shape[dim]
                # sample from the middle if a slice is too big.
                input_slice = slice(diff/2, input.output_shape[dim]-(diff-diff/2))
                input_slicing.append(input_slice)
            self.slices.append(tuple(input_slicing))

        #print
        #print gate.output_shape, input1.output_shape, input2.output_shape
        #print self.slices

    def get_output_shape_for(self, input_shapes):
        return self.smallest_shape

    def get_output_for(self, inputs, **kwargs):
        # take the minimal working slice size, and use that one.

        return inputs[0][self.slices[0]] * inputs[1][self.slices[1]] + (1 - inputs[0][self.slices[0]]) * inputs[2][self.slices[2]]


class PadWithZerosLayer(nn.layers.Layer):
    def __init__(self, incoming, final_size, dimension=1, val=0, **kwargs):
        super(PadWithZerosLayer, self).__init__(incoming, **kwargs)
        self.final_size = final_size
        self.dimension = dimension
        self.val = val

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.dimension] = self.final_size
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        # do nothing if not needed
        if self.input_shape[self.dimension] == self.output_shape[self.dimension]:
            return input

        indices = tuple([slice(0,i) for i in self.input_shape])
        out = T.zeros(self.output_shape)
        return T.set_subtensor(out[indices], input)


def jonas_highway(incoming, num_filters=None,
                  num_conv=3,
                  filter_size=(3,3), pool_size=(2,2), channel=1, axis=(2,3),
                  W=nn.init.Orthogonal(), b=nn.init.Constant(0.0),
                  Wt=nn.init.Orthogonal(), bt=nn.init.Constant(-4.0),
                  nonlinearity=nn.nonlinearities.rectify):

    l_h = incoming
    for _ in xrange(num_conv):
        l_h = ConvolutionOver2DAxisLayer(l_h, num_filters=num_filters,
                                          axis=axis, channel=channel,
                                            filter_size=filter_size,
                                            W=W, b=b,
                                            nonlinearity=nonlinearity)

    l_maxpool = MaxPoolOver2DAxisLayer(l_h, pool_size=pool_size,
                                          stride=pool_size,
                                          axis=axis)

    # reduce the incoming layers size to more or less the remaining size after the
    # previous steps, but with the correct number of channels
    l_maxpool_incoming = MaxPoolOver2DAxisLayer(incoming, pool_size=pool_size,
                                                          stride=pool_size,
                                                          axis=axis)

    l_proc_incoming = PadWithZerosLayer(l_maxpool_incoming,
                                        final_size=num_filters
                                        )

    # gate layer
    l_t = ConvolutionOver2DAxisLayer(l_maxpool_incoming, num_filters=num_filters,
                                        filter_size=filter_size,
                                        W=Wt, b=bt,
                                        nonlinearity=T.nnet.sigmoid)


    return MultiplicativeGatingLayer(gate=l_t, input1=l_maxpool, input2=l_proc_incoming)




"""
def highway_dense(incoming, Wh=nn.init.Orthogonal(), bh=nn.init.Constant(0.0),
              Wt=nn.init.Orthogonal(), bt=nn.init.Constant(-4.0),
              nonlinearity=nn.nonlinearities.rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    # regular layer
    l_h = nn.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh,
                               nonlinearity=nonlinearity)
    # gate layer
    l_t = nn.layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
                               nonlinearity=T.nnet.sigmoid)

    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)


def highway_conv2d(incoming, filter_size,
               Wh=nn.init.Orthogonal(), bh=nn.init.Constant(0.0),
               Wt=nn.init.Orthogonal(), bt=nn.init.Constant(-4.0),
               , **kwargs):
    num_channels = incoming.output_shape[1]
    # regular layer
    l_h = nn.layers.Conv2DLayer(incoming, num_filters=num_channels,
                                filter_size=filter_size,
                                pad='same', W=Wh, b=bh,
                                nonlinearity=nonlinearity)
    # gate layer
    l_t = nn.layers.Conv2DLayer(incoming, num_filters=num_channels,
                                filter_size=filter_size,
                                pad='same', W=Wt, b=bt,
                                nonlinearity=T.nnet.sigmoid)

    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)
"""
