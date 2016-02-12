import theano
import theano.tensor as T
import numpy as np
import nn_heart
import lasagne as nn
import matplotlib.pyplot as plt


def test1():
    slice_mask = T.matrix('mask')  # (batch_size, nslices)
    p0 = T.tensor3('pred')  # (batch_size, nslices, 1)
    t0 = T.matrix('tgt')  # (batch_size, 1)

    mse0 = (p0 - t0.dimshuffle(0, 'x', 1)) ** 2
    mse0 = mse0 * slice_mask.dimshuffle(0, 1, 'x')
    mse0 = T.sum(mse0) / T.sum(slice_mask)
    mse0 = T.sqrt(mse0)

    masked_input = p0 * slice_mask.dimshuffle(0, 1, 'x')
    mean = T.sum(masked_input.flatten(1 + 1), axis=1, keepdims=True) / T.sum(slice_mask, axis=-1, keepdims=True)

    mu_x2 = T.sum(masked_input.flatten(1 + 1) ** 2, axis=1, keepdims=True) / T.sum(slice_mask, axis=-1,
                                                                                   keepdims=True)
    std = T.sqrt(mu_x2 - mean ** 2)

    x_range = T.arange(0, 10).dimshuffle('x', 0)
    mu = T.repeat(mean, 10, axis=1)
    sigma = T.repeat(std, 10, axis=1)
    x = (x_range - mu) / (sigma * T.sqrt(2.) + 1e-16)
    cdf = (T.erf(x) + 1.) / 2.

    f = theano.function([p0, t0, slice_mask], mse0)
    f_mean = theano.function([p0, slice_mask], mean)
    f_std = theano.function([p0, slice_mask], std)
    f_cdf = theano.function([p0, slice_mask], cdf)

    pp0 = np.array([[[5], [3], [7]], [[2], [1], [4]]], dtype='float32')
    tt0 = np.array([[5], [9]], dtype='float32')
    sm = np.array([[1, 1, 0], [1, 0, 0]], dtype='float32')

    print pp0.shape
    print tt0.shape
    print sm.shape

    print f(pp0, tt0, sm)
    print f_mean(pp0, sm)
    print f_std(pp0, sm)
    print f_cdf(pp0, sm)


def test2():
    l_mu = nn.layers.InputLayer((None, 1))
    l_log_sigma = nn.layers.InputLayer((None, 1))

    p0 = nn_heart.NormalCDFLayer(l_mu, l_log_sigma, log=True)
    l_target = nn.layers.InputLayer((None, 1))

    t0_heaviside = nn_heart.heaviside(nn.layers.get_output(l_target))

    crps0 = T.mean((nn.layers.get_output(p0) - t0_heaviside) ** 2, axis=1)

    f_hvs = theano.function([l_target.input_var], t0_heaviside)
    f_cdf = theano.function([l_mu.input_var, l_log_sigma.input_var], nn.layers.get_output(p0))
    f_crps = theano.function([l_mu.input_var, l_log_sigma.input_var, l_target.input_var], crps0)

    mu = np.array([[6], [7], [8]], dtype='float32')
    log_sigma = np.array([[-100], [-100], [-100]], dtype='float32')
    tgt = np.array([[6], [7], [8]], dtype='float32')

    t_hvs = f_hvs(tgt)
    p_cdf = f_cdf(mu, log_sigma)
    print t_hvs[0]
    print p_cdf[0]

    fig = plt.figure()
    x_range = np.arange(600)
    plt.plot(x_range, t_hvs[0])
    plt.plot(x_range, p_cdf[0])
    plt.show()


    print f_crps(mu, log_sigma, tgt)


test2()
