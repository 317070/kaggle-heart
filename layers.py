import lasagne
import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.cuda import dnn
from lasagne.layers.dnn import Conv2DDNNLayer
import theano_printer


class MuLogSigmaErfLayer(lasagne.layers.Layer):

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 600)

    def get_output_for(self, input, **kwargs):
        eps = 1e-7
        x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x',0)
        sigma = T.exp(input[:,1].dimshuffle(0,'x'))

        x = (x_axis - input[:,0].dimshuffle(0,'x')) / (sigma * np.sqrt(2).astype('float32'))
        return (T.erf(x) + 1)/2


class MuSigmaErfLayer(lasagne.layers.Layer):

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 600)

    def get_output_for(self, input, **kwargs):
        eps = 1e-7
        x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x',0)
        sigma = T.clip(T.exp(input[:,1].dimshuffle(0,'x')), eps, 1)
        x = (x_axis - input[:,0].dimshuffle(0,'x')) / (sigma * np.sqrt(2).astype('float32'))
        return (T.erf(x) + 1)/2


class CumSumLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axis=1, **kwargs):
        super(CumSumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return T.extra_ops.cumsum(input, axis=self.axis)


class WideConv2DDNNLayer(Conv2DDNNLayer):

    def __init__(self, incoming, num_filters, filter_size, skip=0, W=lasagne.init.GlorotUniform(), **kwargs):
        super(WideConv2DDNNLayer, self).__init__(incoming,
                                                 num_filters,
                                                 W=W,
                                                 filter_size=filter_size,
                                                 **kwargs)
        self.skip = skip

    def convolve(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)
        else:
            raise NotImplementedError("Only border_mode 'same' has been implemented")

        target_shape = self.input_shape[:2] + (self.input_shape[2]//self.skip, self.skip,) \
                                            + (self.input_shape[3]//self.skip, self.skip,)

        input = input.reshape(target_shape).dimshuffle(0,3,5,1,2,4)
        target_shape = (self.input_shape[0]*self.skip**2,  self.input_shape[1],
                        self.input_shape[2]//self.skip, self.input_shape[3]//self.skip)

        input = input.reshape(target_shape)

        conved = dnn.dnn_conv(img=input,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=border_mode,
                              conv_mode=conv_mode
                              )

        target_shape = (self.input_shape[0], self.skip, self.skip, self.num_filters,
                        self.input_shape[2]//self.skip, self.input_shape[3]//self.skip)

        conved = conved.reshape(target_shape).dimshuffle(0,3,4,1,5,2)

        target_shape = (self.input_shape[0], self.num_filters,
                        self.input_shape[2], self.input_shape[3])

        conved = conved.reshape(target_shape)

        return conved







