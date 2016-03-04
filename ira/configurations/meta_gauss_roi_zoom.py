from collections import namedtuple
import lasagne as nn
import data_iterators
import numpy as np
import theano.tensor as T
import nn_heart
from configuration import subconfig
import utils_heart
from pathfinder import PKL_TRAIN_DATA_PATH, TRAIN_LABELS_PATH, PKL_VALIDATE_DATA_PATH
import utils

caching = None
restart_from_save = None

rng = subconfig().rng
patch_size = subconfig().patch_size
train_transformation_params = subconfig().train_transformation_params
valid_transformation_params = subconfig().valid_transformation_params
test_transformation_params = subconfig().test_transformation_params


batch_size = 8
nbatches_chunk = 2
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.get_train_valid_split(PKL_TRAIN_DATA_PATH)

train_data_iterator = data_iterators.PatientsDataGenerator(data_path=PKL_TRAIN_DATA_PATH,
                                                           batch_size=chunk_size,
                                                           transform_params=train_transformation_params,
                                                           patient_ids=train_valid_ids['train'],
                                                           labels_path=TRAIN_LABELS_PATH,
                                                           slice2roi_path='pkl_train_slice2roi.pkl',
                                                           full_batch=True, random=True, infinite=True, min_slices=5)

valid_data_iterator = data_iterators.PatientsDataGenerator(data_path=PKL_TRAIN_DATA_PATH,
                                                           batch_size=chunk_size,
                                                           transform_params=valid_transformation_params,
                                                           patient_ids=train_valid_ids['valid'],
                                                           labels_path=TRAIN_LABELS_PATH,
                                                           slice2roi_path='pkl_train_slice2roi.pkl',
                                                           full_batch=False, random=False, infinite=False,
                                                           min_slices=5)

test_data_iterator = data_iterators.PatientsDataGenerator(data_path=PKL_VALIDATE_DATA_PATH,
                                                          batch_size=chunk_size,
                                                          transform_params=test_transformation_params,
                                                          slice2roi_path='pkl_validate_slice2roi.pkl',
                                                          full_batch=False, random=False, infinite=False, min_slices=5)

# find maximum number of slices
nslices = max(train_data_iterator.nslices,
              valid_data_iterator.nslices,
              test_data_iterator.nslices)

# set this max for every data iterator
train_data_iterator.nslices = nslices
valid_data_iterator.nslices = nslices
test_data_iterator.nslices = nslices

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 150
learning_rate_schedule = {
    0: 0.0002,
    int(max_nchunks * 0.1): 0.0001,
    int(max_nchunks * 0.3): 0.00008,
    int(max_nchunks * 0.5): 0.00006,
    int(max_nchunks * 0.75): 0.00004,
    int(max_nchunks * 0.9): 0.00001,
    int(max_nchunks * 0.95): 0.000005
}
validate_every = nchunks_per_epoch
save_every = nchunks_per_epoch


def build_model():
    l_in = nn.layers.InputLayer((None, nslices, 30) + patch_size)
    l_in_slice_mask = nn.layers.InputLayer((None, nslices))
    l_in_slice_location = nn.layers.InputLayer((None, nslices, 1))
    l_in_sex_age = nn.layers.InputLayer((None, 2))
    l_ins = [l_in, l_in_slice_mask, l_in_slice_location, l_in_sex_age]

    l_in_rshp = nn.layers.ReshapeLayer(l_in, (-1, 30) + patch_size)  # (batch_size*nslices, 30, h,w)
    submodel = subconfig().build_model(l_in_rshp)

    # ------------------ systole
    l_mu0 = submodel.mu_layers[0]
    l_sigma0 = submodel.sigma_layers[0]

    l_mu0 = nn.layers.ReshapeLayer(l_mu0, (-1, nslices, [1]))
    l_sigma0 = nn.layers.ReshapeLayer(l_sigma0, (-1, nslices, [1]))

    l_volume_mu_sigma0 = nn_heart.JeroenLayer([nn.layers.flatten(l_mu0), nn.layers.flatten(l_sigma0),
                                               l_in_slice_mask, nn.layers.flatten(l_in_slice_location)],
                                              trainable_scale=False)

    l_volume_mu0 = nn.layers.reshape(nn.layers.SliceLayer(l_volume_mu_sigma0, indices=0, axis=-1), ([0], 1))
    l_volume_sigma0 = nn.layers.reshape(nn.layers.SliceLayer(l_volume_mu_sigma0, indices=1, axis=-1), ([0], 1))

    l_cdf0 = nn_heart.NormalCDFLayer(l_volume_mu0, l_volume_sigma0)

    # ------------------ diastole
    l_mu1 = submodel.mu_layers[1]
    l_sigma1 = submodel.sigma_layers[1]

    l_mu1 = nn.layers.ReshapeLayer(l_mu1, (-1, nslices, [1]))
    l_sigma1 = nn.layers.ReshapeLayer(l_sigma1, (-1, nslices, [1]))

    l_volume_mu_sigma1 = nn_heart.JeroenLayer([nn.layers.flatten(l_mu1), nn.layers.flatten(l_sigma1),
                                               l_in_slice_mask, nn.layers.flatten(l_in_slice_location)],
                                              trainable_scale=False)

    l_volume_mu1 = nn.layers.reshape(nn.layers.SliceLayer(l_volume_mu_sigma1, indices=0, axis=-1), ([0], 1))
    l_volume_sigma1 = nn.layers.reshape(nn.layers.SliceLayer(l_volume_mu_sigma1, indices=1, axis=-1), ([0], 1))

    l_cdf1 = nn_heart.NormalCDFLayer(l_volume_mu1, l_volume_sigma1)
    l_outs = [l_cdf0, l_cdf1]
    l_top = nn.layers.MergeLayer(l_outs)

    l_target_mu0 = nn.layers.InputLayer((None, 1))
    l_target_mu1 = nn.layers.InputLayer((None, 1))
    l_targets = [l_target_mu0, l_target_mu1]

    train_params = nn.layers.get_all_params(l_top)
    test_layes = [l_volume_mu_sigma0, l_volume_mu_sigma1]
    mu_layers = [l_volume_mu0, l_volume_mu1]
    sigma_layers = [l_volume_sigma0, l_volume_sigma1]

    return namedtuple('Model',
                      ['l_ins', 'l_outs', 'l_targets', 'l_top', 'train_params', 'submodel', 'test_layers', 'mu_layers',
                       'sigma_layers'])(
        l_ins, l_outs,
        l_targets,
        l_top,
        train_params,
        submodel, test_layes, mu_layers, sigma_layers)


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
    updates = nn.updates.adam(train_loss, model.train_params, learning_rate)
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
