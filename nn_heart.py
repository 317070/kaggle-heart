import lasagne as nn
import theano.tensor as T
import theano
import numpy as np
from lasagne import nonlinearities, init


def heaviside(x):
    return T.arange(0, 600).dimshuffle('x', 0) - T.repeat(x, 600, axis=1) >= 0


def cdf(sample, mu=0, sigma=1, eps=1e-6):
    div = T.sqrt(2) * sigma
    erf_arg = (sample - mu) / div
    return .5 * (1 + T.erf(erf_arg))


class MultLayer(nn.layers.MergeLayer):
    """
    takes elementwise product between 2 layers
    """

    def __init__(self, input1, input2, log=False, **kwargs):
        super(MultLayer, self).__init__([input1, input2], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1]


class NormalCDFLayer(nn.layers.MergeLayer):
    """
    log=True is log_sigma is given else log=False
    """

    def __init__(self, mu, sigma, log=False, **kwargs):
        super(NormalCDFLayer, self).__init__([mu, sigma], **kwargs)
        self.log = log

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 600

    def get_output_for(self, input, **kwargs):
        mu = input[0]
        sigma = input[1]
        if self.log:
            sigma = T.exp(sigma)
        x_range = T.arange(0, 600).dimshuffle('x', 0)
        mu = T.repeat(mu, 600, axis=1)
        sigma = T.repeat(sigma, 600, axis=1)
        x = (x_range - mu) / (sigma * T.sqrt(2.) + 1e-16)
        cdf = (T.erf(x) + 1.) / 2.
        return cdf


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


class MaskedMeanPoolLayer(nn.layers.MergeLayer):
    """
    pools globally across all trailing dimensions beyond the given axis.
    give it a mask
    """

    def __init__(self, incoming, mask, axis, **kwargs):
        super(MaskedMeanPoolLayer, self).__init__([incoming, mask], **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:self.axis] + (1,)

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = inputs[1]
        masked_input = input * mask.dimshuffle(0, 1, 'x')
        return T.sum(masked_input.flatten(self.axis + 1), axis=self.axis, keepdims=True) / T.sum(mask, axis=-1,
                                                                                                 keepdims=True)


class MaskedSTDPoolLayer(nn.layers.MergeLayer):
    """
    pools globally across all trailing dimensions beyond the given axis.
    give it a mask
    """

    def __init__(self, incoming, mask, axis, **kwargs):
        super(MaskedSTDPoolLayer, self).__init__([incoming, mask], **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:self.axis] + (1,)

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = inputs[1]
        masked_input = input * mask.dimshuffle(0, 1, 'x')
        mu_x = T.sum(masked_input.flatten(self.axis + 1), axis=self.axis, keepdims=True) / T.sum(mask, axis=-1,
                                                                                                 keepdims=True)
        mu_x2 = T.sum(masked_input.flatten(self.axis + 1) ** 2, axis=self.axis, keepdims=True) / T.sum(mask, axis=-1,
                                                                                                       keepdims=True)
        return T.sqrt(mu_x2 - mu_x ** 2)


class NonlinearityLayer(nn.layers.Layer):
    def __init__(self, incoming, nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(NonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(input)


class BatchNormLayer(nn.layers.Layer):
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 mode='low_mem', beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha
        self.mode = mode

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized


def batch_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    layer = BatchNormLayer(layer, **kwargs)
    if nonlinearity is not None:
        layer = NonlinearityLayer(layer, nonlinearity)
    return layer


class CumSumLayer(nn.layers.Layer):
    def __init__(self, incoming, axis=1, **kwargs):
        super(CumSumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        result = T.extra_ops.cumsum(input, axis=self.axis)
        return result
