import lasagne as nn
import theano.tensor as T
import numpy as np
from lasagne import nonlinearities
from lasagne.layers.dnn import Conv2DDNNLayer


def lb_softplus(lb=1):
    return lambda x: nn.nonlinearities.softplus(x) + lb


def heaviside(x):
    return T.arange(0, 600).dimshuffle('x', 0) - T.repeat(x, 600, axis=1) >= 0


def crps(predictions, targets_volume):
    targets_heaviside = heaviside(targets_volume)
    return T.mean((predictions - targets_heaviside) ** 2)


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


class ConstantLayer(nn.layers.Layer):
    """
    Makes a layer of constant value the same shape as the given input layer
    """

    def __init__(self, shape_layer, constant=1, **kwargs):
        super(ConstantLayer, self).__init__(shape_layer, **kwargs)
        self.constant = constant

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return T.ones_like(input) * self.constant


class GMMLayer(nn.layers.MergeLayer):
    """
    log=True is log_sigma is given else log=False
    """

    def __init__(self, mu, sigma, w, log=False, **kwargs):
        super(GMMLayer, self).__init__([mu, sigma, w], **kwargs)
        self.log = log

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 600

    def get_output_for(self, input, **kwargs):
        mu = input[0]
        sigma = input[1]
        w = input[2]
        if self.log:
            sigma = T.exp(sigma)

        x_range = T.arange(0, 600).dimshuffle('x', 0, 'x')
        mu = mu.dimshuffle(0, 'x', 1)
        sigma = sigma.dimshuffle(0, 'x', 1)
        x = (x_range - mu) / (sigma * T.sqrt(2.) + 1e-16)
        cdf = (T.erf(x) + 1.) / 2.  # (bs, 600, n_mix)
        cdf = T.sum(cdf * w.dimshuffle(0, 'x', 1), axis=-1)
        return cdf


class RepeatLayer(nn.layers.Layer):
    def __init__(self, incoming, repeats, axis=0, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.repeats = repeats
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(self.axis, self.repeats)
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        shape_ones = [1] * input.ndim
        shape_ones.insert(self.axis, self.repeats)
        ones = T.ones(tuple(shape_ones), dtype=input.dtype)

        pattern = range(input.ndim)
        pattern.insert(self.axis, "x")
        # print shape_ones, pattern
        return ones * input.dimshuffle(*pattern)


class NormalCDFLayer(nn.layers.MergeLayer):
    """
    log=True is log_sigma is given else log=False
    """

    def __init__(self, mu, sigma, sigma_logscale=False, mu_logscale=False, **kwargs):
        super(NormalCDFLayer, self).__init__([mu, sigma], **kwargs)
        self.sigma_logscale = sigma_logscale
        self.mu_logscale = mu_logscale

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 600

    def get_output_for(self, input, **kwargs):
        mu = input[0]
        sigma = input[1]

        if self.sigma_logscale:
            sigma = T.exp(sigma)
        if self.mu_logscale:
            mu = T.exp(mu)

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


class CumSumLayer(nn.layers.Layer):
    def __init__(self, incoming, axis=1, **kwargs):
        super(CumSumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        result = T.extra_ops.cumsum(input, axis=self.axis)
        return result


class NormalisationLayer(nn.layers.Layer):
    def __init__(self, incoming, norm_sum=1.0, allow_negative=False, **kwargs):
        super(NormalisationLayer, self).__init__(incoming, **kwargs)
        self.norm_sum = norm_sum
        self.allow_negative = allow_negative

    def get_output_for(self, input, **kwargs):
        # take the minimal working slice size, and use that one.
        if self.allow_negative:
            inp_low_zero = input - T.min(input, axis=1).dimshuffle(0, 'x')
        else:
            inp_low_zero = input
        return inp_low_zero / T.sum(inp_low_zero, axis=1).dimshuffle(0, 'x') * self.norm_sum


class JeroenLayer(nn.layers.MergeLayer):
    """This layer doesn't overfit; it already knows what to do.
    incomings = [mu_area, sigma_area, is_not_padded, slicelocs]
    output = N x 2 array, with mu = output[:, 0] and sigma = output[:, 1]
    """

    def __init__(self, incomings, trainable_scale=False, **kwargs):
        super(JeroenLayer, self).__init__(incomings, **kwargs)

        self.W_mu = self.add_param(nn.init.Constant(-2.3), (1,), name="W", trainable=trainable_scale)
        self.b_mu = self.add_param(nn.init.Constant(-20.), (1,), name="b", trainable=trainable_scale)

        self.W_sigma = self.add_param(nn.init.Constant(-2.3), (1,), name="W", trainable=trainable_scale)
        self.b_sigma = self.add_param(nn.init.Constant(-20.), (1,), name="b", trainable=trainable_scale)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 2

    def get_output_for(self, inputs, **kwargs):
        mu_area, sigma_area, slice_mask, slicelocs = inputs

        # Rescale input
        mu_area = mu_area * T.exp(self.W_mu[0]) + T.exp(self.b_mu[0])
        sigma_area = sigma_area * T.exp(self.W_sigma[0]) + T.exp(self.b_sigma[0])

        # For each slice pair, compute if both of them are valid
        is_pair = slice_mask[:, :-1] + slice_mask[:, 1:] > 1.5

        # Compute the distance between slices
        h = abs(slicelocs[:, :-1] - slicelocs[:, 1:])
        h /= 10.  # mm to cm

        # Compute mu for each slice pair
        m1 = mu_area[:, :-1]
        m2 = mu_area[:, 1:]
        mu_volumes = (m1 + m2 + T.sqrt(m1 * m2)) * h / 3.0
        mu_volumes *= is_pair

        # Compute sigma for each slice pair
        s1 = sigma_area[:, :-1]
        s2 = sigma_area[:, 1:]
        sigma_volumes = (s1 + s2) * h / 3.0
        sigma_volumes *= is_pair

        # Compute mu and sigma per patient
        mu_volume_patient = T.sum(mu_volumes, axis=1, keepdims=True)
        sigma_volume_patient = T.sqrt(T.sum(sigma_volumes ** 2, axis=1, keepdims=True))

        # Concat and return
        return T.concatenate([mu_volume_patient, sigma_volume_patient], axis=1)