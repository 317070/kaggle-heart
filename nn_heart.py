import lasagne
import theano.tensor as T


class NormalCDFLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, log_sigma, **kwargs):
        super(NormalCDFLayer, self).__init__([mu, log_sigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0]

    def get_output_for(self, input, **kwargs):
        mu, log_sigma = input
        sigma = T.exp(log_sigma)
        x_range = T.arange(0, 600).dimshuffle('x', 0)
        # TODO check
        x = (x_range - mu) / (sigma * T.sqrt(2.))
        cdf = (T.erf(x) + 1.) / 2.
        return cdf
