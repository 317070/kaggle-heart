from collections import namedtuple
import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
import data_iterators
import numpy as np
import theano.tensor as T
from functools import partial
import nn_heart
import utils_heart
from pathfinder import PKL_TRAIN_DATA_PATH, TRAIN_LABELS_PATH, PKL_VALIDATE_DATA_PATH
import data
import utils

caching = None
restart_from_save = None
rng = np.random.RandomState(42)
patch_size = (64, 64)
mm_patch_size = (128, 128)
train_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': mm_patch_size,
    'rotation_range': (-180, 180),
    'mask_roi': False,
    'translation_range_x': (-10, 10),
    'translation_range_y': (-10, 10),
    'shear_range': (0, 0),
    'roi_scale_range': (1.2, 1.5),
    'do_flip': (False, True),
    'zoom_range': (1 / 1.5, 1.5),
    'sequence_shift': False
}

valid_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': mm_patch_size,
    'mask_roi': False
}

test_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': mm_patch_size,
    'rotation_range': (-180, 180),
    'mask_roi': False,
    'translation_range_x': (-10, 10),
    'translation_range_y': (-10, 10),
    'shear_range': (0, 0),
    'roi_scale_range': (1.2, 1.5),
    'do_flip': (False, True),
    'zoom_range': (1., 1.),
    'sequence_shift': False
}

data_prep_fun = data.transform_norm_rescale_after

batch_size = 32
nbatches_chunk = 13
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.get_train_valid_split(PKL_TRAIN_DATA_PATH)

train_data_iterator = data_iterators.SliceNormRescaleDataGenerator(data_path=PKL_TRAIN_DATA_PATH,
                                                                   batch_size=chunk_size,
                                                                   transform_params=train_transformation_params,
                                                                   patient_ids=train_valid_ids['train'],
                                                                   labels_path=TRAIN_LABELS_PATH,
                                                                   slice2roi_path='pkl_train_slice2roi.pkl',
                                                                   full_batch=True, random=True, infinite=True,
                                                                   view='2ch',
                                                                   data_prep_fun=data_prep_fun)

valid_data_iterator = data_iterators.SliceNormRescaleDataGenerator(data_path=PKL_TRAIN_DATA_PATH,
                                                                   batch_size=chunk_size,
                                                                   transform_params=valid_transformation_params,
                                                                   patient_ids=train_valid_ids['valid'],
                                                                   labels_path=TRAIN_LABELS_PATH,
                                                                   slice2roi_path='pkl_train_slice2roi.pkl',
                                                                   full_batch=False, random=False, infinite=False,
                                                                   view='2ch',
                                                                   data_prep_fun=data_prep_fun)

test_data_iterator = data_iterators.SliceNormRescaleDataGenerator(data_path=PKL_VALIDATE_DATA_PATH,
                                                                  batch_size=chunk_size,
                                                                  transform_params=test_transformation_params,
                                                                  slice2roi_path='pkl_validate_slice2roi.pkl',
                                                                  full_batch=False, random=False, infinite=False,
                                                                  view='2ch',
                                                                  data_prep_fun=data_prep_fun)

nchunks_per_epoch = max(1, train_data_iterator.nsamples / chunk_size)
max_nchunks = nchunks_per_epoch * 500
learning_rate_schedule = {
    0: 0.0001,
    int(max_nchunks * 0.5): 0.00008,
    int(max_nchunks * 0.6): 0.00004,
    int(max_nchunks * 0.8): 0.00001,
    int(max_nchunks * 0.9): 0.000005
}
validate_every = nchunks_per_epoch
save_every = nchunks_per_epoch

conv3 = partial(Conv2DDNNLayer,
                stride=(1, 1),
                pad="same",
                filter_size=(3, 3),
                nonlinearity=nn.nonlinearities.very_leaky_rectify,
                b=nn.init.Constant(0.1),
                W=nn.init.Orthogonal("relu"))

max_pool = partial(MaxPool2DDNNLayer,
                   pool_size=(2, 2),
                   stride=(2, 2))


def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, 30) + patch_size) if not l_in else l_in

    l = conv3(l_in, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)

    l = max_pool(l)

    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)

    l = max_pool(l)

    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)

    l = max_pool(l)

    l_d01 = nn.layers.DenseLayer(l, num_units=1024, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1), nonlinearity=None)
    l_d01 = nn.layers.FeaturePoolLayer(l_d01, pool_size=2)

    l_d02 = nn.layers.DenseLayer(nn.layers.dropout(l_d01), num_units=1024, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1), nonlinearity=None)
    l_d02 = nn.layers.FeaturePoolLayer(l_d02, pool_size=2)

    mu0 = nn.layers.DenseLayer(nn.layers.dropout(l_d02), num_units=1, W=nn.init.Orthogonal(),
                               b=nn.init.Constant(50), nonlinearity=nn_heart.lb_softplus())
    sigma0 = nn.layers.DenseLayer(nn.layers.dropout(l_d02), num_units=1, W=nn.init.Orthogonal(),
                                  b=nn.init.Constant(10), nonlinearity=nn_heart.lb_softplus())
    l_cdf0 = nn_heart.NormalCDFLayer(mu0, sigma0, sigma_logscale=False, mu_logscale=False)
    # ---------------------------------------------------------------

    l_d11 = nn.layers.DenseLayer(l, num_units=1024, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1), nonlinearity=None)
    l_d11 = nn.layers.FeaturePoolLayer(l_d11, pool_size=2)

    l_d12 = nn.layers.DenseLayer(nn.layers.dropout(l_d11), num_units=1024, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1), nonlinearity=None)
    l_d12 = nn.layers.FeaturePoolLayer(l_d12, pool_size=2)

    mu1 = nn.layers.DenseLayer(nn.layers.dropout(l_d12), num_units=1, W=nn.init.Orthogonal(),
                               b=nn.init.Constant(100), nonlinearity=nn_heart.lb_softplus())
    sigma1 = nn.layers.DenseLayer(nn.layers.dropout(l_d12), num_units=1, W=nn.init.Orthogonal(),
                                  b=nn.init.Constant(10), nonlinearity=nn_heart.lb_softplus())
    l_cdf1 = nn_heart.NormalCDFLayer(mu1, sigma1, sigma_logscale=False, mu_logscale=False)


    l_outs = [l_cdf0, l_cdf1]
    l_top = nn.layers.MergeLayer(l_outs)

    l_target_mu0 = nn.layers.InputLayer((None, 1))
    l_target_mu1 = nn.layers.InputLayer((None, 1))
    l_targets = [l_target_mu0, l_target_mu1]
    mu_layers = [mu0, mu1]
    sigma_layers = [sigma0, sigma1]

    return namedtuple('Model', ['l_ins', 'l_outs', 'l_targets', 'l_top', 'mu_layers', 'sigma_layers'])([l_in], l_outs,
                                                                                                       l_targets, l_top,
                                                                                                       mu_layers,
                                                                                                       sigma_layers)


def build_objective(model, deterministic=False):
    p0 = nn.layers.get_output(model.l_outs[0], deterministic=deterministic)
    t0 = nn.layers.get_output(model.l_targets[0])
    t0_heaviside = nn_heart.heaviside(t0)

    crps0 = T.mean((p0 - t0_heaviside) ** 2)

    p1 = nn.layers.get_output(model.l_outs[1], deterministic=deterministic)
    t1 = nn.layers.get_output(model.l_targets[1])
    t1_heaviside = nn_heart.heaviside(t1)

    crps1 = T.mean((p1 - t1_heaviside) ** 2)

    return 0.5 * (crps0 + crps1)


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_top), learning_rate)
    return updates


def get_mean_validation_loss(batch_predictions, batch_targets):
    return [0, 0]


def get_mean_crps_loss(batch_predictions, batch_targets, batch_ids):
    nbatches = len(batch_predictions)
    npredictions = len(batch_predictions[0])

    crpss = []
    for i in xrange(npredictions):
        p, t = [], []
        for j in xrange(nbatches):
            p.append(batch_predictions[j][i])
            t.append(batch_targets[j][i])
        p, t = np.vstack(p), np.vstack(t)
        target_cdf = utils_heart.heaviside_function(t)
        crpss.append(np.mean((p - target_cdf) ** 2))
    return crpss


def get_avg_patient_predictions(batch_predictions, batch_patient_ids, mean):
    return utils_heart.get_patient_average_cdf_predictions(batch_predictions, batch_patient_ids, mean)
