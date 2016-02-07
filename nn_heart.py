import lasagne as nn
import theano.tensor as T
import theano
import numpy as np


def cdf(sample, mu=0, sigma=1, eps=1e-6):
    div = T.sqrt(2) * sigma
    erf_arg = (sample - mu) / div
    return .5 * (1 + T.erf(erf_arg))


class NormalCDFLayer(nn.layers.MergeLayer):
    def __init__(self, mu, log_sigma, **kwargs):
        super(NormalCDFLayer, self).__init__([mu, log_sigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 600

    def get_output_for(self, input, **kwargs):
        mu, log_sigma = input
        sigma = T.exp(log_sigma)
        x_range = T.arange(0, 600).dimshuffle('x', 0)
        mu = T.repeat(mu, 600, axis=1)
        sigma = T.repeat(sigma, 600, axis=1)
        x = (x_range - mu) / (sigma * T.sqrt(2.) + 1e-16)
        cdf = (T.erf(x) + 1.) / 2.
        return cdf


class NormalizationLayer(nn.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return (input - T.mean(input, axis=[-2, -1], keepdims=True)) / T.std(input, axis=[-2, -1], keepdims=True)


class AttentionLayer(nn.layers.Layer):
    def __init__(self, incoming, u=nn.init.GlorotUniform(), **kwargs):
        super(AttentionLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.u = self.add_param(u, (num_inputs, 1), name='u')

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_output_for(self, input, **kwargs):
        a = T.nnet.softmax(T.dot(input, self.u)[:, :, 0])
        return T.sum(a[:, :, np.newaxis] * input, axis=1)


class MaskedGlobalMeanPoolLayer(nn.layers.MergeLayer):
    """
    pools globally across all trailing dimensions beyond the given axis.
    give it a mask
    """

    def __init__(self, incoming, mask, axis, **kwargs):
        super(MaskedGlobalMeanPoolLayer, self).__init__([incoming, mask], **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:self.axis] + (1,)

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = inputs[1]
        masked_input = input * mask.dimshuffle(0, 1, 'x')
        return T.sum(masked_input.flatten(self.axis + 1), axis=self.axis, keepdims=True) / T.sum(mask, axis=-1,
                                                                                                 keepdims=True)
