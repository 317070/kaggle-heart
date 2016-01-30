import lasagne as nn
import theano.tensor as T


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
