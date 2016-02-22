from collections import namedtuple
import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
import data_iterators
import numpy as np
import theano.tensor as T
from functools import partial
import utils_heart
import nn_heart

caching = None

restart_from_save = None
rng = np.random.RandomState(42)
patch_size = (64, 64)
train_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': (128, 128),
    'mask_roi': False,
    'rotation_range': (-180, 180),
    'translation_range_x': (-5, 10),
    'translation_range_y': (-10, 5),
    'shear_range': (0, 0),
    'roi_scale_range': (0.8, 1.2),
    'do_flip': True,
    'sequence_shift': False
}

valid_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': (128, 128)
}

batch_size = 32
nbatches_chunk = 13
chunk_size = batch_size * nbatches_chunk

train_data_iterator = data_iterators.SliceNormRescaleDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/train',
                                                                   batch_size=chunk_size,
                                                                   transform_params=train_transformation_params,
                                                                   labels_path='/data/dsb15_pkl/train.csv',
                                                                   slice2roi_path='pkl_train_slice2roi.pkl',
                                                                   full_batch=True, random=True, infinite=True,
                                                                   view='2ch')

valid_data_iterator = data_iterators.SliceNormRescaleDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/valid',
                                                                   batch_size=chunk_size,
                                                                   transform_params=valid_transformation_params,
                                                                   labels_path='/data/dsb15_pkl/train.csv',
                                                                   slice2roi_path='pkl_train_slice2roi.pkl',
                                                                   full_batch=False, random=False, infinite=False,
                                                                   view='2ch')

test_data_iterator = data_iterators.SliceNormRescaleDataGenerator(data_path='/data/dsb15_pkl/pkl_validate',
                                                                  batch_size=chunk_size,
                                                                  transform_params=train_transformation_params,
                                                                  slice2roi_path='pkl_validate_slice2roi.pkl',
                                                                  full_batch=False, random=False, infinite=False,
                                                                  view='2ch')

nchunks_per_epoch = max(1, train_data_iterator.nsamples / chunk_size)
max_nchunks = nchunks_per_epoch * 500
learning_rate_schedule = {
    0: 0.0001,
    int(max_nchunks * 0.6): 0.00008,
    int(max_nchunks * 0.7): 0.00004,
    int(max_nchunks * 0.8): 0.00002,
    int(max_nchunks * 0.9): 0.00001
}
validate_every = nchunks_per_epoch
save_every = nchunks_per_epoch

conv3 = partial(Conv2DDNNLayer,
                stride=(1, 1),
                pad="same",
                filter_size=(3, 3),
                nonlinearity=nn.nonlinearities.rectify,
                b=nn.init.Constant(0.1),
                W=nn.init.Orthogonal("relu"))

max_pool = partial(MaxPool2DDNNLayer,
                   pool_size=(2, 2),
                   stride=(2, 2))


def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, 30) + patch_size) if not l_in else l_in

    l = conv3(l_in, num_filters=64)
    l = conv3(l, num_filters=64)

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

    l_d01 = nn.layers.DenseLayer(l, num_units=512, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1))
    l_d02 = nn.layers.DenseLayer(nn.layers.dropout(l_d01, p=0.5), num_units=512, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1))

    l_sm0 = nn.layers.DenseLayer(nn.layers.dropout(l_d02, p=0.5), num_units=600, b=nn.init.Constant(0.1),
                                 nonlinearity=nn.nonlinearities.softmax)
    l_sm0 = nn_heart.NormalisationLayer(nn.layers.dropout(l_sm0, p=0.5))
    l_cdf0 = nn_heart.CumSumLayer(l_sm0)

    # ---------------------------------------------------------------

    l_d11 = nn.layers.DenseLayer(l, num_units=512, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1))
    l_d12 = nn.layers.DenseLayer(nn.layers.dropout(l_d11, p=0.5), num_units=512, W=nn.init.Orthogonal("relu"),
                                 b=nn.init.Constant(0.1))

    l_sm1 = nn.layers.DenseLayer(nn.layers.dropout(l_d12, p=0.5), num_units=600, b=nn.init.Constant(0.1),
                                 nonlinearity=nn.nonlinearities.softmax)
    l_sm1 = nn_heart.NormalisationLayer(nn.layers.dropout(l_sm1, p=0.5))
    l_cdf1 = nn_heart.CumSumLayer(l_sm1)

    l_outs = [l_cdf0, l_cdf1]
    l_top = nn.layers.MergeLayer(l_outs)

    l_target_mu0 = nn.layers.InputLayer((None, 1))
    l_target_mu1 = nn.layers.InputLayer((None, 1))
    l_targets = [l_target_mu0, l_target_mu1]
    dense_layers = [l_d01, l_d02, l_d11, l_d12]

    return namedtuple('Model', ['l_ins', 'l_outs', 'l_targets', 'l_top', 'dense_layers'])([l_in], l_outs, l_targets,
                                                                                          l_top, dense_layers)


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
