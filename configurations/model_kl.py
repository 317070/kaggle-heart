from collections import namedtuple
import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from nn_heart import NormalizationLayer, NormalCDFLayer
import data_iterators
import numpy as np
import theano
import theano.tensor as T
import utils

restart_from_save = None
rng = np.random.RandomState(42)
patch_size = (64, 64)
train_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': (-16, 16),
    'translation_range': (-8, 8),
    'shear_range': (0, 0)
}

valid_transformation_params = {
    'patch_size': (64, 64),
    'rotation_range': None,
    'translation_range': None,
    'shear_range': None
}

batch_size = 4
max_niter = 15000
learning_rate_schedule = {
    0: 0.0001,
    5000: 0.0001,
    10000: 0.00003
}
validate_every = 50
save_every = 50
l2_weight = 1e-3

train_data_iterator = data_iterators.SlicesVolumeDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted_small/train',
                                                               batch_size=batch_size,
                                                               transform_params=train_transformation_params,
                                                               labels_path='/data/dsb15_pkl/train.csv', full_batch=True,
                                                               random=True, infinite=True)

valid_data_iterator = data_iterators.SlicesVolumeDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted_small/train',
                                                               batch_size=batch_size,
                                                               transform_params=valid_transformation_params,
                                                               labels_path='/data/dsb15_pkl/train.csv',
                                                               full_batch=False,
                                                               random=False)

test_data_iterator = data_iterators.SlicesVolumeDataGenerator(data_path='/data/dsb15_pkl/pkl_validate',
                                                              batch_size=batch_size,
                                                              transform_params=valid_transformation_params,
                                                              full_batch=False,
                                                              random=False)


def build_model():
    l_in = nn.layers.InputLayer((None, 30) + patch_size)

    l_norm = NormalizationLayer(l_in)

    l = Conv2DDNNLayer(l_norm, num_filters=64, filter_size=(3, 3),
                       W=nn.init.Orthogonal('relu'),
                       b=nn.init.Constant(0.1),
                       pad='same')
    l = Conv2DDNNLayer(l, num_filters=64, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1),
                       pad="same")

    l = MaxPool2DDNNLayer(l, pool_size=(2, 2))

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1),
                       pad="same")
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1),
                       pad="same")

    l = MaxPool2DDNNLayer(l, pool_size=(2, 2))

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1), pad='same')
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1), pad='same')
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2))
    # --------------------------------------------------------------
    l_d0 = nn.layers.DenseLayer(nn.layers.dropout(l, p=0.5), num_units=512, nonlinearity=nn.nonlinearities.tanh)
    l_mu0 = nn.layers.DenseLayer(l_d0, num_units=1, nonlinearity=nn.nonlinearities.identity)
    l_log_sigma0 = nn.layers.DenseLayer(l_d0, num_units=1, nonlinearity=nn.nonlinearities.identity)

    # ---------------------------------------------------------------
    l_d1 = nn.layers.DenseLayer(nn.layers.dropout(l, p=0.5), num_units=512, nonlinearity=nn.nonlinearities.tanh)
    l_mu1 = nn.layers.DenseLayer(l_d1, num_units=1, nonlinearity=nn.nonlinearities.identity)
    l_log_sigma1 = nn.layers.DenseLayer(l_d1, num_units=1, nonlinearity=nn.nonlinearities.identity)

    l_outs = [l_mu0, l_log_sigma0, l_mu1, l_log_sigma1]
    l_top = nn.layers.MergeLayer(l_outs)

    l_target_mu0 = nn.layers.InputLayer((None, 1))
    l_target_mu1 = nn.layers.InputLayer((None, 1))

    l_targets = [l_target_mu0, l_target_mu1]
    return namedtuple('Model', ['l_ins', 'l_outs', 'l_targets', 'l_top', 'l_params'])([l_in], l_outs, l_targets, l_top,
                                                                                      [l_mu0, l_log_sigma0, l_mu1,
                                                                                       l_log_sigma1])


def build_objective(model, deterministic=False):
    mu0 = nn.layers.get_output(model.l_outs[0], deterministic=deterministic)
    log_sigma0 = nn.layers.get_output(model.l_outs[1], deterministic=deterministic)
    mu1 = nn.layers.get_output(model.l_outs[2], deterministic=deterministic)
    log_sigma1 = nn.layers.get_output(model.l_outs[3], deterministic=deterministic)

    mu0_target = nn.layers.get_output(model.l_targets[0])
    mu1_target = nn.layers.get_output(model.l_targets[1])

    d_kl0 = T.mean(
        0.5 * (mu0_target ** 2 - 2. * mu0_target * mu0 + mu0 ** 2 + T.exp(log_sigma0) - 1. - log_sigma0))
    d_kl1 = T.mean(
        0.5 * (mu1_target ** 2 - 2. * mu1_target * mu1 + mu1 ** 2 + T.exp(log_sigma1) - 1. - log_sigma1))

    return d_kl0 + d_kl1


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_top), learning_rate)
    return updates


def get_mean_crps_loss(batch_predictions, batch_targets):
    nbatches = len(batch_predictions)
    npredictions = len(batch_predictions[0])

    crpss = []
    for i in xrange(npredictions):
        p, t = [], []
        for j in xrange(nbatches):
            p.append(batch_predictions[j][i])
            t.append(batch_targets[j][i])
        p, t = np.vstack(p), np.vstack(t)

        crpss.append(np.mean((p - t) ** 2))
    return np.mean(crpss)


def get_mean_validation_loss(batch_predictions, batch_targets):
    return get_mean_crps_loss(batch_predictions, batch_targets)
