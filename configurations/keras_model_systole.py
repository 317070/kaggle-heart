from collections import namedtuple
import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from nn_heart import NormalizationLayer
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

batch_size = 32
max_niter = 500
learning_rate_schedule = {
    0: 0.0001
}
validate_every = 20
save_every = 20
l2_weight = 1e-3

train_data_iterator = data_iterators.SlicesDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/train',
                                                         batch_size=batch_size,
                                                         transform_params=train_transformation_params,
                                                         labels_path='/data/dsb15_pkl/train.csv', full_batch=True,
                                                         random=True, infinite=True)

valid_data_iterator = data_iterators.SlicesDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/valid',
                                                         batch_size=batch_size,
                                                         transform_params=valid_transformation_params,
                                                         labels_path='/data/dsb15_pkl/train.csv', full_batch=False,
                                                         random=False)

test_data_iterator = data_iterators.SlicesDataGenerator(data_path='/data/dsb15_pkl/pkl_validate', batch_size=batch_size,
                                                        transform_params=valid_transformation_params, full_batch=False,
                                                        random=False)


def build_model():
    l_in = nn.layers.InputLayer((None, 30) + patch_size)

    l_norm = NormalizationLayer(l_in)
    l_out0, l_d0 = model(l_norm)
    l_out1, l_d1 = model(l_norm)

    l_targets = [nn.layers.InputLayer((None, 1)), nn.layers.InputLayer((None, 1))]
    l_top = nn.layers.MergeLayer([l_out0, l_out1])
    return namedtuple('Model', ['l_ins', 'l_outs', 'l_targets', 'l_top', 'regularizable_layers'])([l_in],
                                                                                                  [l_out0, l_out1],
                                                                                                  [l_targets], l_top,
                                                                                                  [l_d0, l_d1])


def model(l_in):
    l = Conv2DDNNLayer(l_in, num_filters=64, filter_size=(3, 3),
                       W=nn.init.Orthogonal('relu'),
                       b=nn.init.Constant(0.1),
                       pad='same')
    l = Conv2DDNNLayer(l, num_filters=64, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1),
                       pad="valid")

    l = nn.layers.PadLayer(l, width=(1, 1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    l = nn.layers.DropoutLayer(l, p=0.25)

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1),
                       pad="same")
    l = Conv2DDNNLayer(l, num_filters=96, filter_size=(3, 3),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1),
                       pad="valid")

    l = nn.layers.PadLayer(l, width=(1, 1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    l = nn.layers.DropoutLayer(l, p=0.25)

    # ---------------------------------------------------------------
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(2, 2),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1))
    l = Conv2DDNNLayer(l, num_filters=128, filter_size=(2, 2),
                       W=nn.init.Orthogonal("relu"),
                       b=nn.init.Constant(0.1))
    l = MaxPool2DDNNLayer(l, pool_size=(2, 2), stride=(2, 2))
    l = nn.layers.DropoutLayer(l, p=0.25)

    # --------------------------------------------------------------

    l = nn.layers.FlattenLayer(l)
    l_d1 = nn.layers.DenseLayer(l, num_units=1024, W=nn.init.Orthogonal('relu'), b=nn.init.Constant(0.1))
    l_out = nn.layers.DenseLayer(nn.layers.dropout(l_d1, p=0.5), num_units=1, W=nn.init.Orthogonal('relu'),
                                 b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.identity)

    return l_out, l_d1


def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_outs[0], deterministic=deterministic)
    targets = nn.layers.get_output(model.l_targets[0])

    if model.regularizable_layers:
        regularization_dict = {}
        for l in model.regularizable_layers:
            regularization_dict[l] = l2_weight
        l2_penalty = nn.regularization.regularize_layer_params_weighted(regularization_dict,
                                                                        nn.regularization.l2)
    else:
        l2_penalty = 0.0
    return T.sqrt(T.mean(predictions - targets) ** 2) + l2_penalty


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
        sigma = utils.rmse(p, t)
        crpss.append(utils.crps(p, t, sigma))
    return crpss


def get_mean_validation_loss(batch_predictions, batch_targets):
    nbatches = len(batch_predictions)
    npredictions = len(batch_predictions[0])

    losses = []
    for i in xrange(npredictions):
        x, y = [], []
        for j in xrange(nbatches):
            x.append(batch_predictions[j][i])
            y.append(batch_targets[j][i])
        x, y = np.vstack(x), np.vstack(y)
        print x.shape
        print y.shape
        losses.append(np.sqrt(np.mean((x - y) ** 2)))

    if npredictions == 1:
        losses = losses[0]
    return losses
