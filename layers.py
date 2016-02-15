import lasagne
from lasagne.layers import Conv1DLayer
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
        # This needs to be clipped to avoid NaN's!
        sigma = T.exp(T.clip(input[:,1].dimshuffle(0,'x'), -10, 10))
        #theano_printer.print_me_this("sigma", sigma)
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


class MuConstantSigmaErfLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma=0.0, **kwargs):
        super(MuConstantSigmaErfLayer, self).__init__(incoming, **kwargs)
        eps = 1e-7
        if sigma>=eps:
            self.sigma = theano.shared(np.float32(sigma)).dimshuffle('x', 'x')
        else:
            self.sigma = theano.shared(np.float32(eps)).dimshuffle('x', 'x')


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 600)

    def get_output_for(self, input, **kwargs):
        x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x', 0)
        x = (x_axis - input[:,0].dimshuffle(0, 'x')) / (self.sigma * np.sqrt(2).astype('float32'))
        return (T.erf(x) + 1)/2


class CumSumLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axis=1, **kwargs):
        super(CumSumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        result = T.extra_ops.cumsum(input, axis=self.axis)
        # theano_printer.print_me_this("result", result)
        return result


class ScaleLayer(lasagne.layers.MergeLayer):
    def __init__(self, input, scale, **kwargs):
        incomings = [input, scale]
        super(ScaleLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        # take the minimal working slice size, and use that one.
        return inputs[0] * T.shape_padright(inputs[1], n_ones=inputs[0].ndim-inputs[1].ndim)


class NormalisationLayer(lasagne.layers.Layer):
    def __init__(self, incoming, norm_sum=1.0, **kwargs):
        super(NormalisationLayer, self).__init__(incoming, **kwargs)
        self.norm_sum = norm_sum

    def get_output_for(self, input, **kwargs):
        # take the minimal working slice size, and use that one.
        inp_low_zero = input #- T.min(input)
        return inp_low_zero / T.sum(inp_low_zero) * self.norm_sum


class WideConv2DDNNLayer(Conv2DDNNLayer):

    def __init__(self, incoming, num_filters, filter_size, skip=0, **kwargs):
        super(WideConv2DDNNLayer, self).__init__(incoming,
                                                 num_filters,
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



