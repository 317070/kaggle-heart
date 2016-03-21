"""Library implementing various Lasagne layers.
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T

from lasagne.layers import Conv1DLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from theano.sandbox.cuda import dnn

import theano_printer
import utils


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

    def get_output_for(self, input, eps=1e-7, **kwargs):
        x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x',0)
        sigma = input[:,1].dimshuffle(0,'x')
        x = (x_axis - input[:,0].dimshuffle(0,'x')) / (sigma * np.sqrt(2).astype('float32'))
        return (T.erf(x) + 1.0)/2.0


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


class LogicalNotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(LogicalNotLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # take the minimal working slice size, and use that one.
        return 1 - input


class NormalisationLayer(lasagne.layers.Layer):
    """Layer which normalises the input over the first axis.

    Normalisation is achieved by simply dividing by the sum.
    """
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


class JeroenLayer(lasagne.layers.MergeLayer):
    """This layer doesn't overfit; it already knows what to do.

    Estimates the volume between slices using a truncated cone approximation.

    incomings = [mu_area, sigma_area, is_not_padded, slicelocs]
    output = N x 2 array, with mu = output[:, 0] and sigma = output[:, 1]
    """
    def __init__(self, incomings, rescale_input=1.0, **kwargs):
        super(JeroenLayer, self).__init__(incomings, **kwargs)
        self.rescale_input = rescale_input

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 2)

    def get_output_for(self, inputs, **kwargs):
        mu_area, sigma_area, is_not_padded, slicelocs = inputs

        # Rescale input
        mu_area = mu_area / self.rescale_input
        sigma_area = sigma_area / self.rescale_input
        
        # For each slice pair, compute if both of them are valid
        is_pair_not_padded = is_not_padded[:, :-1] + is_not_padded[:, 1:] > 1.5
        
        # Compute the distance between slices
        h = abs(slicelocs[:, :-1] - slicelocs[:, 1:])
        
        # Compute mu for each slice pair
        m1 = mu_area[:, :-1]
        m2 = mu_area[:, 1:]
        eps = 1e-2
        mu_volumes = (m1 + m2 + T.sqrt(T.clip(m1*m2, eps, utils.maxfloat))) * h / 3.0
        mu_volumes = mu_volumes * is_pair_not_padded
        
        # Compute sigma for each slice pair
        s1 = sigma_area[:, :-1]
        s2 = sigma_area[:, 1:]
        sigma_volumes = h*(s1 + s2) / 3.0
        sigma_volumes = sigma_volumes * is_pair_not_padded
        
        # Compute mu and sigma per patient
        mu_volume_patient = T.sum(mu_volumes, axis=1)
        sigma_volume_patient = T.sqrt(T.clip(T.sum(sigma_volumes**2, axis=1), eps, utils.maxfloat))

        # Concat and return
        return T.concatenate([
            mu_volume_patient.dimshuffle(0, 'x'),
            sigma_volume_patient.dimshuffle(0, 'x')], axis=1)


class JeroenLayerDists(lasagne.layers.MergeLayer):
    """JeroenLayer using better distances.

    This layer expects the distances between slices as an input, so computing or
    estimating the distances offloaded to other modules. This allows
    exploration of alternative ways to compute slice distances.
    """

    def __init__(self, incomings, rescale_input=1.0, **kwargs):
        super(JeroenLayerDists, self).__init__(incomings, **kwargs)
        self.rescale_input = rescale_input

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 2)

    def get_output_for(self, inputs, **kwargs):
        mu_area, sigma_area, is_not_padded, slicedists = inputs

        # Rescale input
        mu_area = mu_area / self.rescale_input
        sigma_area = sigma_area / self.rescale_input

        # For each slice pair, compute if both of them are valid
        is_pair_not_padded = is_not_padded[:, :-1] + is_not_padded[:, 1:] > 1.5

        # Compute the distance between slices
        h = slicedists[:, :-1]

        # Compute mu for each slice pair
        m1 = mu_area[:, :-1]
        m2 = mu_area[:, 1:]
        eps = 1e-2
        mu_volumes = (m1 + m2 + T.sqrt(T.clip(m1*m2, eps, utils.maxfloat))) * h / 3.0
        mu_volumes = mu_volumes * is_pair_not_padded

        # Compute sigma for each slice pair
        s1 = sigma_area[:, :-1]
        s2 = sigma_area[:, 1:]
        sigma_volumes = h*(s1 + s2) / 3.0
        sigma_volumes = sigma_volumes * is_pair_not_padded

        # Compute mu and sigma per patient
        mu_volume_patient = T.sum(mu_volumes, axis=1)
        sigma_volume_patient = T.sqrt(T.clip(T.sum(sigma_volumes**2, axis=1), eps, utils.maxfloat))

        # Concat and return
        return T.concatenate([
            mu_volume_patient.dimshuffle(0, 'x'),
            sigma_volume_patient.dimshuffle(0, 'x')], axis=1)


class JeroenLayerDiscs(lasagne.layers.MergeLayer):
    """JeroenLayers using discs instead of truncated cones.
    """
    def __init__(self, incomings, rescale_input=1.0, **kwargs):
        super(JeroenLayerDiscs, self).__init__(incomings, **kwargs)
        self.rescale_input = rescale_input

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 2)

    def get_output_for(self, inputs, **kwargs):
        mu_area, sigma_area, is_not_padded, slicedists = inputs

        # Rescale input
        mu_area = mu_area / self.rescale_input
        sigma_area = sigma_area / self.rescale_input

        # For each slice pair, compute if both of them are valid
        is_pair_not_padded = is_not_padded[:, :-1] + is_not_padded[:, 1:] > 1.5

        # Compute the distance between slices
        h = slicedists[:, :-1]

        # Compute mu for each slice pair
        m1 = mu_area[:, :-1]
        m2 = mu_area[:, 1:]
        eps = 1e-2
        mu_volumes = (m1 + m2) * h / 2.0
        mu_volumes = mu_volumes * is_pair_not_padded

        # Compute sigma for each slice pair
        s1 = sigma_area[:, :-1]
        s2 = sigma_area[:, 1:]
        sigma_volumes = h * T.sqrt(s1**2 + s2**2 + eps) / 2.0
        sigma_volumes = sigma_volumes * is_pair_not_padded

        # Compute mu and sigma per patient
        mu_volume_patient = T.sum(mu_volumes, axis=1)
        sigma_volume_patient = T.sqrt(T.clip(T.sum(sigma_volumes**2, axis=1), eps, utils.maxfloat))

        # Concat and return
        return T.concatenate([
            mu_volume_patient.dimshuffle(0, 'x'),
            sigma_volume_patient.dimshuffle(0, 'x')], axis=1)


class WeightedMeanLayer(lasagne.layers.MergeLayer):
    def __init__(self, weights, incomings, **kwargs):
        super(WeightedMeanLayer, self).__init__([weights]+incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        conc = T.concatenate([i.dimshuffle(0,1,'x') for i in inputs[1:]], axis=2)
        weights = T.nnet.softmax(inputs[0])
        r    = conc * weights.dimshuffle(0,'x',1)
        result = T.sum(r, axis=2)
        return result



class IncreaseCertaintyLayer(lasagne.layers.MergeLayer):
    def __init__(self, weight, incoming, **kwargs):
        super(IncreaseCertaintyLayer, self).__init__([weight,incoming], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        """

        :param inputs[0]: (batch, 600)
        :param inputs[1]: (batch, 1)
        :return:
        """
        result = (T.erf( T.erfinv( T.clip(inputs[1].dimshuffle(0,'x',1)*2-1, -1+3e-8, 1-3e-8) ) * inputs[0].dimshuffle(0,1,'x') )+1)/2
        return result[:,0,:]


class TrainableScaleLayer(lasagne.layers.Layer):

    def __init__(self, incoming, scale=lasagne.init.Constant(1), trainable=True, **kwargs):
        super(TrainableScaleLayer, self).__init__(incoming, **kwargs)

        # create scales parameter, ignoring all dimensions in shared_axes
        shape = []

        self.scale = self.add_param(
            scale, shape, 'scale', regularizable=False, trainable=trainable)

    def get_output_for(self, input, **kwargs):
        return input * self.scale.dimshuffle('x', 'x')


class RelativeLocationLayer(lasagne.layers.Layer):
    """Layer implementing a function mapping outermost values to 1 and innermost
    values to 0.
    """
    def __init__(self, slicelocations, **kwargs):
        super(RelativeLocationLayer, self).__init__(slicelocations, **kwargs)


    def get_output_for(self, slicelocations, **kwargs):
        x = slicelocations - T.min(slicelocations, axis=1).dimshuffle(0, 'x')
        return abs(x * 2.0 / T.max(x, axis=1).dimshuffle(0, 'x') - 1.0)


class RepeatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, repeats, axis=0, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.repeats = repeats
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(self.axis, self.repeats)
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return repeat(input, self.repeats, self.axis)


def repeat(input, repeats, axis):
    shape_ones = [1]*input.ndim
    shape_ones.insert(axis, repeats)
    ones = T.ones(tuple(shape_ones), dtype=input.dtype)

    pattern = range(input.ndim)
    pattern.insert(axis, "x")
    # print shape_ones, pattern
    return ones * input.dimshuffle(*pattern)




class IraLayer(lasagne.layers.MergeLayer):
    """Layer estimating the volume of the heart using 2CH and 4CH images.

    The volume is estimated by stacking elliptical discs. 
    For each 'row' in the 2ch and 4ch image, it expects the mu and sigma of the
    expected width of the heart chamber in that row.
    """
    def __init__(self, l_4ch_mu, l_4ch_sigma, l_2ch_mu, l_2ch_sigma, trainable_scale=False, **kwargs):
        super(IraLayer, self).__init__([l_4ch_mu, l_4ch_sigma, l_2ch_mu, l_2ch_sigma], **kwargs)

        if trainable_scale:
            self.W_mu = self.add_param(lasagne.init.Constant(-2.3), (1,), name="W")
            self.b_mu = self.add_param(lasagne.init.Constant(-20.), (1,), name="b")

            self.W_sigma = self.add_param(lasagne.init.Constant(-2.3), (1,), name="W")
            self.b_sigma = self.add_param(lasagne.init.Constant(-20.), (1,), name="b")

        self.trainable_scale = trainable_scale

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 600, 2

    def get_output_for(self, inputs, **kwargs):
        eps = 1e-7
        mu_a, sigma_a, mu_b, sigma_b = inputs

        # Rescale input
        if self.trainable_scale:
            mu_a = mu_a * T.exp(self.W_mu[0]) + T.exp(self.b_mu[0])
            sigma_a = sigma_a * T.exp(self.W_sigma[0]) + T.exp(self.b_sigma[0])

            mu_b = mu_b * T.exp(self.W_mu[0]) + T.exp(self.b_mu[0])
            sigma_b = sigma_b * T.exp(self.W_sigma[0]) + T.exp(self.b_sigma[0])

        # Compute the distance between slices
        h = 0.1  # mm to cm

        # Compute mu for each slice pair
        mu_volumes = mu_a * mu_b * h
        #  (batch, time, height)

        # Compute sigma for each slice pair
        var_a = sigma_a ** 2
        var_b = sigma_b ** 2
        var_volumes = (var_a * var_b + var_a * mu_b ** 2 + var_b * mu_a ** 2) * h ** 2
        #  (batch, time, height)

        # Compute mu and sigma per patient

        mu_volume_patient = np.pi / 4. * T.sum(mu_volumes, axis=2)
        #  (batch, time)

        sigma_volume_patient = np.pi / 4. * T.sqrt(T.clip(T.sum(var_volumes, axis=2), eps, utils.maxfloat))
        sigma_volume_patient = T.clip(sigma_volume_patient, eps, utils.maxfloat)
        #  (batch, time)

        x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x', 'x', 0)
        x = (x_axis - mu_volume_patient.dimshuffle(0, 1, 'x')) / (sigma_volume_patient.dimshuffle(0, 1, 'x') )
        prediction_matrix = (T.erf(x) + 1)/2

        #  (batch, time, 600)

        # max because distribution of smaller one will lie higher
        l_systole = T.max(prediction_matrix, axis=1)
        l_diastole = T.min(prediction_matrix, axis=1)
        #  (batch, 600)

        return T.concatenate([l_systole.dimshuffle(0, 1, 'x'),
                              l_diastole.dimshuffle(0, 1, 'x')], axis=2)


class SumGaussLayer(lasagne.layers.MergeLayer):
    """Sum of two gaussians

    Input = 3d
    """
    def __init__(self, incomings, **kwargs):
        super(SumGaussLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        if (
            not input_shapes[0] == input_shapes[1]
            or not len(input_shapes[0]) == 3
            or not input_shapes[0][-1] == 2):
            raise ValueError("Invalid input shapes %s" % str(input_shapes))

        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        musigma_1, musigma_2 = inputs

        mu_1 = musigma_1[:, :, 0]
        sigma_1 = musigma_1[:, :, 1]
        mu_2 = musigma_2[:, :, 0]
        sigma_2 = musigma_2[:, :, 1]

        mu_res = mu_1 + mu_2
        sigma_res = np.sqrt(sigma_1**2 + sigma_2**2)

        return T.concatenate([
            mu_res.dimshuffle(0,1,'x'),
            sigma_res.dimshuffle(0,1,'x')], axis=2)


class IraLayerNoTime(lasagne.layers.MergeLayer):
    """Similar to IraLayer, but without handling the time dimension.
    """
    def __init__(self, l_4ch_musigma, l_2ch_musigma, **kwargs):
        super(IraLayerNoTime, self).__init__([l_4ch_musigma, l_2ch_musigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], input_shapes[0][2]

    def get_output_for(self, inputs, **kwargs):
        eps = 1e-7
        musigma_1, musigma_2 = inputs
        mu_1 = musigma_1[:, :, 0]
        sigma_1 = musigma_1[:, :, 1]
        mu_2 = musigma_2[:, :, 0]
        sigma_2 = musigma_2[:, :, 1]

        # Compute the distance between slices
        h = 0.1  # mm to cm

        # Compute mu and sigma for each slice pair disc
        mu_volumes = mu_1 * mu_2 * h * np.pi / 4.0
        sigma_volumes =  T.sqrt(T.clip(sigma_1**2 * sigma_2**2 + sigma_1**2 * mu_2**2 + sigma_2**2 * mu_1**2, eps, utils.maxfloat)) * h * np.pi / 4.0

        # Compute mu and sigma per patient
        mu_volume_patient = T.sum(mu_volumes, axis=1)

        sigma_volume_patient = T.sqrt(T.clip(T.sum(sigma_volumes**2, axis=1), eps, utils.maxfloat))
        sigma_volume_patient = T.clip(sigma_volume_patient, eps, utils.maxfloat)

        return T.concatenate([
            mu_volume_patient.dimshuffle(0, 'x'),
            sigma_volume_patient.dimshuffle(0, 'x')], axis=1)


class ArgmaxAndMaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, mode='max', **kwargs):
        super(ArgmaxAndMaxLayer, self).__init__(incoming, **kwargs)
        if not mode in ('max', 'min', 'mean'):
            raise ValueError('invalid mode')
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        if not len(input_shape) == 3:
            raise ValueError('Require input to be a 3D tensor.')
        if not input_shape[2] == 2:
            raise ValueError('Requires inputs last dimension to be 2')
        return (input_shape[0], input_shape[2])

    def get_output_for(self, input, **kwargs):
        if self.mode in ('max', 'min'):
            reductor = T.argmax if self.mode == 'max' else T.argmin
            indices1 = T.arange(input.shape[0])
            indices2 = reductor(input[:, :, 0], axis=1)
            return input[indices1, indices2]
        elif self.mode == 'mean':
            return T.mean(input, axis=1)


class IntegrateAreaLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma_mode='scale', sigma_scale=None, **kwargs):
        super(IntegrateAreaLayer, self).__init__(incoming, **kwargs)
        if not sigma_mode in ('scale', 'smart',):
            raise ValueError('invalid mode')
        self.sigma_mode = sigma_mode
        self.sigma_scale = sigma_scale

    def get_output_shape_for(self, input_shape):
        if not len(input_shape) > 2:
            raise ValueError('Require input to hae at least 3 dimensions.')
        return tuple(list(input_shape[:-2]) + [2])

    def get_output_for(self, input, **kwargs):
        # compute mu
        mu = input.sum(axis=-1).sum(axis=-1, keepdims=True)
        # compute sigma
        if self.sigma_mode == 'scale':
            sigma = mu * self.sigma_scale
        elif self.sigma_mode == 'smart':
            eps = 1e-3
            sigma = T.sqrt((input * (1-input)).sum(axis=-1).sum(axis=-1, keepdims=True) + eps)
        else:
            raise NotImplementedError('sigma mode not implemented')
        # concatenate and return
        return T.concatenate([mu, sigma], axis=-1)


class SelectWithAttentionLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(SelectWithAttentionLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        shape_musigma, shape_att = input_shapes
        if not len(shape_musigma) == 3:
            raise ValueError('Require input to be a 3D tensor.')
        if not shape_musigma[-1] == 2:
            raise ValueError('Last dimension of input should be 2.')
        if not len(shape_att) == 2:
            raise ValueError('Requires attention dimension to be 2')
        if not shape_musigma[:2] == shape_att:
            print shape_musigma, shape_att
            raise ValueError('First dimensions should be the same')
        return (shape_musigma[0], shape_musigma[2])

    def get_output_for(self, inputs, **kwargs):
        musigma, att = inputs
        mu = (musigma[:, :, 0] * att).sum(axis=1, keepdims=True)
        sigma = (musigma[:, :, 1] * att).sum(axis=1, keepdims=True)
        return T.concatenate((mu, sigma), axis=1)
