import sys
import numpy as np
import theano
from itertools import izip
import lasagne as nn
import utils
import buffering
import utils_heart
from configuration import config, set_configuration, set_subconfiguration
from pathfinder import METADATA_PATH

if not (len(sys.argv) < 3):
    sys.exit("Usage: predict.py <metadata_path>")

metadata_path = sys.argv[1]
metadata_dir = utils.get_dir_path('train', METADATA_PATH)
metadata = utils.load_pkl(metadata_dir + '/%s' % metadata_path)
config_name = metadata['configuration']
if 'subconfiguration' in metadata:
    set_subconfiguration(metadata['subconfiguration'])

set_configuration(config_name)

# predictions paths
prediction_dir = utils.get_dir_path('predictions', METADATA_PATH)
prediction_path = prediction_dir + "/%s.pkl" % metadata['experiment_id']
prediction_mu_std_path = prediction_dir + "/%s_mu_sigma.pkl" % metadata['experiment_id']

print "Build model"
model = config().build_model()
all_layers = nn.layers.get_all_layers(model.l_top)
all_params = nn.layers.get_all_params(model.l_top)
num_params = nn.layers.count_params(model.l_top)
print '  number of parameters: %d' % num_params
nn.layers.set_all_param_values(model.l_top, metadata['param_values'])

xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
givens_in = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_in[l_in.input_var] = x

iter_test_det = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_outs],
                                givens=givens_in, on_unused_input='warn')

iter_mu = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.mu_layers], givens=givens_in,
                          on_unused_input='warn')
iter_sigma = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.sigma_layers],
                             givens=givens_in, on_unused_input='warn')

print ' generating predictions for the validation set'
valid_data_iterator = config().valid_data_iterator

batch_predictions, batch_targets, batch_ids = [], [], []
mu_predictions, sigma_predictions = [], []
for xs_batch_valid, ys_batch_valid, ids_batch in buffering.buffered_gen_threaded(
        valid_data_iterator.generate()):
    for x_shared, x in zip(xs_shared, xs_batch_valid):
        x_shared.set_value(x)

    batch_targets.append(ys_batch_valid)
    batch_predictions.append(iter_test_det())
    batch_ids.append(ids_batch)
    mu_predictions.append(iter_mu())
    sigma_predictions.append(iter_sigma())

pid2mu = utils_heart.get_patient_normparam_prediction(mu_predictions, batch_ids)
pid2sigma = utils_heart.get_patient_normparam_prediction(sigma_predictions, batch_ids)
valid_pid2musigma = {}
for pid in pid2mu.iterkeys():
    valid_pid2musigma[pid] = {'mu': pid2mu[pid], 'sigma': pid2sigma[pid]}

valid_avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean='geometric')
patient_targets = utils_heart.get_patient_average_heaviside_predictions(batch_targets, batch_ids)

assert valid_avg_patient_predictions.viewkeys() == patient_targets.viewkeys()
crpss_sys, crpss_dst = [], []
for id in valid_avg_patient_predictions.iterkeys():
    crpss_sys.append(utils_heart.crps(valid_avg_patient_predictions[id][0], patient_targets[id][0]))
    crpss_dst.append(utils_heart.crps(valid_avg_patient_predictions[id][1], patient_targets[id][1]))
    print id, 0.5 * (crpss_sys[-1] + crpss_dst[-1]), crpss_sys[-1], crpss_dst[-1]

crps0, crps1 = np.mean(crpss_sys), np.mean(crpss_dst)

print 'Valid crps average: ', 0.5 * (crps0 + crps1)

print ' generating predictions for the train set'
train_data_iterator = config().train_data_iterator
train_data_iterator.transformation_params = config().valid_transformation_params
train_data_iterator.full_batch = False
train_data_iterator.random = False
train_data_iterator.infinite = False

batch_predictions, batch_targets, batch_ids = [], [], []
mu_predictions, sigma_predictions = [], []
for xs_batch_valid, ys_batch_valid, ids_batch in buffering.buffered_gen_threaded(
        train_data_iterator.generate()):
    for x_shared, x in zip(xs_shared, xs_batch_valid):
        x_shared.set_value(x)

    batch_targets.append(ys_batch_valid)
    batch_predictions.append(iter_test_det())
    batch_ids.append(ids_batch)
    mu_predictions.append(iter_mu())
    sigma_predictions.append(iter_sigma())

pid2mu = utils_heart.get_patient_normparam_prediction(mu_predictions, batch_ids)
pid2sigma = utils_heart.get_patient_normparam_prediction(sigma_predictions, batch_ids)
train_pid2musigma = {}
for pid in pid2mu.iterkeys():
    train_pid2musigma[pid] = {'mu': pid2mu[pid], 'sigma': pid2sigma[pid]}

train_avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean='geometric')
patient_targets = utils_heart.get_patient_average_heaviside_predictions(batch_targets, batch_ids)

assert train_avg_patient_predictions.viewkeys() == patient_targets.viewkeys()
crpss_sys, crpss_dst = [], []
for id in train_avg_patient_predictions.iterkeys():
    crpss_sys.append(utils_heart.crps(train_avg_patient_predictions[id][0], patient_targets[id][0]))
    crpss_dst.append(utils_heart.crps(train_avg_patient_predictions[id][1], patient_targets[id][1]))
    print id, 0.5 * (crpss_sys[-1] + crpss_dst[-1]), crpss_sys[-1], crpss_dst[-1]

crps0, crps1 = np.mean(crpss_sys), np.mean(crpss_dst)

print 'Train crps average: ', 0.5 * (crps0 + crps1)

# test
print ' generating predictions for the test set'
test_data_iterator = config().test_data_iterator
test_data_iterator.transformation_params = config().valid_transformation_params

batch_predictions, batch_ids = [], []
mu_predictions, sigma_predictions = [], []
for xs_batch_test, _, ids_batch in buffering.buffered_gen_threaded(test_data_iterator.generate()):
    for x_shared, x in zip(xs_shared, xs_batch_test):
        x_shared.set_value(x)
    batch_predictions.append(iter_test_det())
    batch_ids.append(ids_batch)
    mu_predictions.append(iter_mu())
    sigma_predictions.append(iter_sigma())

test_avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean='geometric')
pid2mu = utils_heart.get_patient_normparam_prediction(mu_predictions, batch_ids)
pid2sigma = utils_heart.get_patient_normparam_prediction(sigma_predictions, batch_ids)
test_pid2musigma = {}
for pid in pid2mu.iterkeys():
    test_pid2musigma[pid] = {'mu': pid2mu[pid], 'sigma': pid2sigma[pid]}

predictions = {'train': train_avg_patient_predictions,
               'valid': valid_avg_patient_predictions,
               'test': test_avg_patient_predictions}

predictions_mu_std = {'train': train_pid2musigma,
                      'valid': valid_pid2musigma,
                      'test': test_pid2musigma}

utils.save_pkl(predictions, prediction_path)
utils.save_pkl(predictions_mu_std, prediction_mu_std_path)
print ' predictions saved to %s' % prediction_path
