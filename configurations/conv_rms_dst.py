from collections import namedtuple
import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from nn_heart import NormalizationLayer
import data_iterators
import numpy as np
import theano.tensor as T
import utils
from collections import defaultdict

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
    'patch_size': patch_size,
    'rotation_range': None,
    'translation_range': None,
    'shear_range': None
}

batch_size = 32
nbatches_chunk = 16
chunk_size = batch_size * nbatches_chunk

train_data_iterator = data_iterators.SlicesDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/train',
                                                         batch_size=chunk_size,
                                                         transform_params=train_transformation_params,
                                                         labels_path='/data/dsb15_pkl/train.csv',
                                                         full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.SlicesDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/valid',
                                                         batch_size=chunk_size,
                                                         transform_params=valid_transformation_params,
                                                         labels_path='/data/dsb15_pkl/train.csv',
                                                         full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 150
learning_rate_schedule = {
    0: 0.0001,
    int(max_nchunks * 0.25): 0.00007,
    int(max_nchunks * 0.5): 0.00003,
    int(max_nchunks * 0.75): 0.00001,
}
validate_every = nchunks_per_epoch
save_every = nchunks_per_epoch
l2_weight = 0.0005


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
    l_d1 = nn.layers.DenseLayer(nn.layers.dropout(l, p=0.25), num_units=1024, W=nn.init.Orthogonal("relu"),
                                b=nn.init.Constant(0.1))
    l_mu1 = nn.layers.DenseLayer(nn.layers.dropout(l_d1, p=0.5), num_units=1, nonlinearity=nn.nonlinearities.identity)

    l_outs = [l_mu1, l_mu1]
    l_top = l_mu1

    l_target_mu0 = nn.layers.InputLayer((None, 1))
    l_target_mu1 = nn.layers.InputLayer((None, 1))
    l_targets = [l_target_mu0, l_target_mu1]

    return namedtuple('Model', ['l_ins', 'l_outs', 'l_targets', 'l_top', 'regularizable_layers'])([l_in], l_outs,
                                                                                                  l_targets, l_top,
                                                                                                  [l_d1])


def build_objective(model, deterministic=False):
    p1 = nn.layers.get_output(model.l_outs[1], deterministic=deterministic)
    t1 = nn.layers.get_output(model.l_targets[1])

    if model.regularizable_layers:
        regularization_dict = {}
        for l in model.regularizable_layers:
            regularization_dict[l] = l2_weight
        l2_penalty = nn.regularization.regularize_layer_params_weighted(regularization_dict,
                                                                        nn.regularization.l2)
    else:
        l2_penalty = 0.0

    return T.sqrt(T.mean((p1 - t1) ** 2)) + l2_penalty


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_top), learning_rate)
    return updates


def get_mean_validation_loss(batch_predictions, batch_targets):
    nbatches = len(batch_predictions)
    x, y = [], []
    for j in xrange(nbatches):
        x.append(batch_predictions[j][1])
        y.append(batch_targets[j][1])
    x, y = np.vstack(x), np.vstack(y)
    return np.sqrt(np.mean((x - y) ** 2))


def get_mean_crps_loss(batch_predictions, batch_targets, batch_ids):
    nbatches = len(batch_predictions)

    patient_ids = []
    for i in xrange(nbatches):
        patient_ids += batch_ids[i]

    patient2idxs = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        patient2idxs[pid].append(i)

    # collect predictions over batches
    p, t = [], []
    for j in xrange(nbatches):
        p.append(batch_predictions[j][1])
        t.append(batch_targets[j][1])
    p, t = np.vstack(p), np.vstack(t)

    # collect crps over patients
    patient_crpss = []
    for patient_id, patient_idxs in patient2idxs.iteritems():
        prediction_cdf = utils.heaviside_function(p[patient_idxs])
        avg_prediction_cdf = np.mean(prediction_cdf, axis=0)
        target_cdf = utils.heaviside_function(t[patient_idxs])[0]
        patient_crpss.append(utils.crps(avg_prediction_cdf, target_cdf))

    return np.mean(patient_crpss)
