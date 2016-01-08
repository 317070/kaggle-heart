import numpy as np
import lasagne
import theano
import theano.tensor as T

"""
If the input is a segmentation with probabilities, this means that getting the distribution of the output is a Poisson binomial distribution.
It is pretty much unfeasible to calculate this one exactly.

It therefore has to be approximated.
"""

class GaussianApproximationVolumeLayer(lasagne.layers.Layer):
    """
    Gaussian approximation: https://en.wikipedia.org/wiki/Poisson_binomial_distribution#Mean_and_variance
    """
    def __init__(self, incoming, **kwargs):
        super(GaussianApproximationVolumeLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        # (batch, time, 600)
        return (input_shape[0], input_shape[1], 600)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 3:
            # input: (batch, time, axis, verti, horiz)
            # needs: (batch, time, pixels)
            input = input.flatten(ndim=3)

        mu = T.sum(input, axis=2).dimshuffle(0,1,'x')
        sigma = T.sum(input * (1-input), axis=2).dimshuffle(0,1,'x')
        x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x','x',0)
        x = (x_axis - mu) / sigma
        return (T.erf(x) + 1)/2

