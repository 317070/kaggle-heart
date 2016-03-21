"""Library providing some custom Lasagne layers.
"""

import lasagne
import theano
import theano.tensor as T
import numpy as np

from lasagne.layers.dnn import Conv2DDNNLayer, Conv3DDNNLayer, MaxPool2DDNNLayer, MaxPool3DDNNLayer
from lasagne.utils import as_tuple
from lasagne.layers import Conv1DLayer, MaxPool1DLayer, Layer


class ConvolutionOver2DAxisLayer(Conv2DDNNLayer):
    def __init__(self, incoming, num_filters, filter_size, channel=1, axis=(2,3), **kwargs):
        super(ConvolutionOver2DAxisLayer, self).__init__(incoming,
                                                         num_filters,
                                                         filter_size=filter_size,
                                                         check_shape=False,
                                                         **kwargs)
        self.axis = axis
        self.channel = channel

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        # axis channel shrinks
        for i in xrange(2):
            shape[self.axis[i]] = lasagne.layers.conv.conv_output_length(shape[self.axis[i]], self.filter_size[i], self.stride[i], self.pad[i])
        # filter channel changes
        shape[self.channel] = self.num_filters
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            raise NotImplementedError("untie_biases has not been implemented")
        else:
            shuffle = ['x']*len(self.input_shape)
            shuffle[self.channel] = 0
            activation = conved + self.b.dimshuffle(tuple(shuffle))
        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        dimshuffle = range(len(self.input_shape))
        dimshuffle.remove(self.channel)
        for axis in self.axis:
            dimshuffle.remove(axis)
        dimshuffle.append(self.channel)
        for axis in self.axis:
            dimshuffle.append(axis)

        input = input.dimshuffle(*dimshuffle).reshape((-1,
                                                      self.input_shape[self.channel],
                                                      self.input_shape[self.axis[0]],
                                                      self.input_shape[self.axis[1]],
                                                      ))

        conved = super(ConvolutionOver2DAxisLayer, self).convolve(input, **kwargs)

        output_shape = list(self.input_shape)
        output_shape = [i for j, i in enumerate(output_shape) if j not in [self.channel, self.axis[0], self.axis[1]]]

        conv_output_shape = self.get_output_shape_for(self.input_shape)

        conved = conved.reshape(tuple(output_shape)+
                                (self.num_filters,
                                  conv_output_shape[self.axis[0]],
                                  conv_output_shape[self.axis[1]],
                                 )
                                )

        reverse_dimshuffle = [dimshuffle.index(i) for i in xrange(len(self.input_shape))]
        conved = conved.dimshuffle(*reverse_dimshuffle)

        return conved


class MaxPoolOver2DAxisLayer(MaxPool2DDNNLayer):
    def __init__(self, incoming, pool_size, axis=(2,3), **kwargs):
        super(MaxPoolOver2DAxisLayer, self).__init__(incoming, pool_size, check_shape=False, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        # axis channels shrink
        for i in xrange(2):
            shape[self.axis[i]] = lasagne.layers.pool.pool_output_length(shape[self.axis[i]],
                                                                         pool_size=self.pool_size[i],
                                                                         stride=self.stride[i],
                                                                         pad=self.pad[i],
                                                                         ignore_border=True
                                                                         )
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        dimshuffle = range(len(self.input_shape))
        for axis in self.axis:
            dimshuffle.remove(axis)
        for axis in self.axis:
            dimshuffle.append(axis)

        input = input.dimshuffle(*dimshuffle).reshape((-1,
                                                      1,
                                                      self.input_shape[self.axis[0]],
                                                      self.input_shape[self.axis[1]],
                                                      ))

        pooled = super(MaxPoolOver2DAxisLayer, self).get_output_for(input, **kwargs)
        output_shape = list(self.input_shape)
        output_shape = [i for j, i in enumerate(output_shape) if j not in [self.axis[0], self.axis[1]]]

        conv_output_shape = self.get_output_shape_for(self.input_shape)

        pooled = pooled.reshape(tuple(output_shape)+
                                (conv_output_shape[self.axis[0]],
                                  conv_output_shape[self.axis[1]],
                                 )
                                )

        reverse_dimshuffle = [dimshuffle.index(i) for i in xrange(len(self.input_shape))]
        pooled = pooled.dimshuffle(*reverse_dimshuffle)
        return pooled



class ConvolutionOver3DAxisLayer(Conv3DDNNLayer):
    def __init__(self, incoming, num_filters, filter_size, channel=1, axis=(2,3,4), **kwargs):
        super(ConvolutionOver3DAxisLayer, self).__init__(incoming,
                                                 num_filters,
                                                 filter_size=filter_size,
                                                 check_shape=False,
                                                 **kwargs)
        self.axis = axis
        self.channel = channel

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        # axis channel shrinks
        for i in xrange(3):
            shape[self.axis[i]] = lasagne.layers.conv.conv_output_length(shape[self.axis[i]], self.filter_size[i], self.stride[i], self.pad[i])
        # filter channel changes
        shape[self.channel] = self.num_filters
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            raise NotImplementedError("untie_biases has not been implemented")
        else:
            shuffle = ['x']*len(self.input_shape)
            shuffle[self.channel] = 0
            activation = conved + self.b.dimshuffle(tuple(shuffle))
        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        dimshuffle = range(len(self.input_shape))
        dimshuffle.remove(self.channel)
        for axis in self.axis:
            dimshuffle.remove(axis)
        dimshuffle.append(self.channel)
        for axis in self.axis:
            dimshuffle.append(axis)

        input = input.dimshuffle(*dimshuffle).reshape((-1,
                                                      self.input_shape[self.channel],
                                                      self.input_shape[self.axis[0]],
                                                      self.input_shape[self.axis[1]],
                                                      self.input_shape[self.axis[2]],
                                                      ))
        conved = super(ConvolutionOver3DAxisLayer, self).convolve(input, **kwargs)
        output_shape = list(self.input_shape)
        output_shape = [i for j, i in enumerate(output_shape) if j not in [self.channel, self.axis[0], self.axis[1], self.axis[2]]]
        conv_output_shape = self.get_output_shape_for(self.input_shape)
        output_shape = tuple(output_shape)+(self.num_filters,
                                  conv_output_shape[self.axis[0]],
                                  conv_output_shape[self.axis[1]],
                                  conv_output_shape[self.axis[2]],
                                 )
        conved = conved.reshape(output_shape)
        reverse_dimshuffle = [dimshuffle.index(i) for i in xrange(len(self.input_shape))]
        conved = conved.dimshuffle(*reverse_dimshuffle)
        return conved



class MaxPoolOver3DAxisLayer(MaxPool3DDNNLayer):
    def __init__(self, incoming, pool_size, axis=(2,3,4), **kwargs):
        super(MaxPoolOver3DAxisLayer, self).__init__(incoming, pool_size, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        # axis channels shrink
        for i in xrange(3):
            shape[self.axis[i]] = lasagne.layers.pool.pool_output_length(shape[self.axis[i]],
                                                                         pool_size=self.pool_size[i],
                                                                         stride=self.stride[i],
                                                                         pad=self.pad[i],
                                                                         ignore_border=True
                                                                         )
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        dimshuffle = range(len(self.input_shape))
        for axis in self.axis:
            dimshuffle.remove(axis)
        for axis in self.axis:
            dimshuffle.append(axis)
        input = input.dimshuffle(*dimshuffle).reshape((-1,
                                                      1,
                                                      self.input_shape[self.axis[0]],
                                                      self.input_shape[self.axis[1]],
                                                      self.input_shape[self.axis[2]],
                                                      ))

        pooled = super(MaxPoolOver3DAxisLayer, self).get_output_for(input, **kwargs)
        output_shape = list(self.input_shape)
        output_shape = [i for j, i in enumerate(output_shape) if j not in self.axis]

        conv_output_shape = self.get_output_shape_for(self.input_shape)
        output_shape = tuple(output_shape)+ (conv_output_shape[self.axis[0]],
                                  conv_output_shape[self.axis[1]],
                                  conv_output_shape[self.axis[2]],
                                 )
        pooled = pooled.reshape(output_shape)
        reverse_dimshuffle = [dimshuffle.index(i) for i in xrange(len(self.input_shape))]
        pooled = pooled.dimshuffle(*reverse_dimshuffle)
        return pooled


class ConvolutionOverAxisLayer(ConvolutionOver2DAxisLayer):

    def __init__(self, incoming, num_filters, filter_size, channel=1, axis=(2,), **kwargs):
        assert channel != 0, "using batch as either axis or channel is not supported"
        if axis[0] == len(incoming.output_shape)-1:
            axis = (axis[0], len(incoming.output_shape)-2)
        else:
            axis = (axis[0], len(incoming.output_shape)-1)
        filter_size = (filter_size[0], 1)
        super(ConvolutionOverAxisLayer, self).__init__(incoming,
                                                     num_filters,
                                                     axis=axis,
                                                     channel=channel,
                                                     filter_size=filter_size,
                                                     **kwargs)


class MaxPoolOverAxisLayer(MaxPoolOver2DAxisLayer):

    def __init__(self, incoming, pool_size, axis=(2,), **kwargs):
        if axis[0] == len(incoming.output_shape)-1:
            axis = (len(incoming.output_shape)-2, axis[0])
        else:
            axis = (len(incoming.output_shape)-1, axis[0])
        pool_size = (1, pool_size[0])
        super(MaxPoolOverAxisLayer, self).__init__(incoming,
                                             pool_size=pool_size,
                                             axis=axis,
                                             **kwargs)


class FixedScaleLayer(Layer):
    """Layer which simply scales the input by a fixed factor.
    """
    def __init__(self, incoming, scale=1, **kwargs):
        super(FixedScaleLayer, self).__init__(incoming, **kwargs)
        self.scale = scale
      
    def get_output_for(self, input, **kwargs):
        return input * self.scale


def FixedConstantLayer(constant):
    """Function creating an InputLayer which outputs a fixed scalar.
    """
    theano_var = theano.shared(constant)
    # Add 0 to the theano var, to prevent it from returning a cudandarray later
    theano_var += theano.shared(np.array(0.0, dtype='float32'))
    return lasagne.layers.InputLayer(
        constant.shape, input_var=theano_var) 