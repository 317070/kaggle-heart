import sys
import numpy as np
import theano
from itertools import izip
import lasagne as nn
import cPickle as pickle
import utils
import buffering
import utils_heart
from configuration import config, set_configuration, set_subconfiguration
from pathfinder import METADATA_PATH, PREDICTIONS_PATH
import logger

NUM_PATIENTS = 1140

if not (3 <= len(sys.argv) <= 5):
    sys.exit("Usage: predict.py <config_name> <n_tta_iterations> <average: arithmetic|geometric>")

config_name = sys.argv[1]
n_tta_iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
mean = sys.argv[3] if len(sys.argv) >= 4 else 'geometric'

print 'Make %s tta predictions for %s set using %s mean' % (n_tta_iterations, "valid and test", mean)

metadata_dir = utils.get_dir_path('train', METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)
metadata = utils.load_pkl(metadata_path)
assert config_name == metadata['configuration']
if 'subconfiguration' in metadata:
    set_subconfiguration(metadata['subconfiguration'])
set_configuration(config_name)

# predictions paths
jonas_prediction_path = PREDICTIONS_PATH + '/ira_%s.pkl' % config().__name__
prediction_dir = utils.get_dir_path('predictions', METADATA_PATH)
valid_prediction_path = prediction_dir + "/%s-%s-%s-%s.pkl" % (
    metadata['experiment_id'], 'valid', n_tta_iterations, mean)
test_prediction_path = prediction_dir + "/%s-%s-%s-%s.pkl" % (metadata['experiment_id'], 'test', n_tta_iterations, mean)

# submissions paths
submission_dir = utils.get_dir_path('submissions', METADATA_PATH)
submission_path = submission_dir + "/%s-%s-%s-%s.csv" % (metadata['experiment_id'], 'test', n_tta_iterations, mean)

# logs
logs_dir = utils.get_dir_path('logs', METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/ira_predictions_%s.log' % metadata['experiment_id'])
sys.stderr = sys.stdout


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
                                givens=givens_in, on_unused_input='ignore')

predictions = [{"patient": i + 1,
                "systole": np.zeros((0, 600)),
                "diastole": np.zeros((0, 600))
                } for i in xrange(NUM_PATIENTS)]

# validation
valid_data_iterator = config().valid_data_iterator
if n_tta_iterations > 1:
    valid_data_iterator.transformation_params = config().train_transformation_params
    valid_data_iterator.transformation_params['zoom_range'] = (1., 1.)

print 'valid transformation params'
print valid_data_iterator.transformation_params

print
print 'n valid: %d' % valid_data_iterator.nsamples

batch_predictions, batch_targets, batch_ids = [], [], []
for i in xrange(n_tta_iterations):
    print i,
    sys.stdout.flush()
    for xs_batch_valid, ys_batch_valid, ids_batch in buffering.buffered_gen_threaded(
            valid_data_iterator.generate()):
        for x_shared, x in zip(xs_shared, xs_batch_valid):
            x_shared.set_value(x)
        batch_targets.append(ys_batch_valid)
        batch_predictions.append(iter_test_det())
        batch_ids.append(ids_batch)

for (systole_predictions, diastole_predictions), patient_ids in zip(batch_predictions, batch_ids):
    for systole_prediction, diastole_prediction, patient_id in zip(systole_predictions, diastole_predictions,
                                                                   patient_ids):
        patient_data = predictions[patient_id - 1]
        assert patient_data['patient'] == patient_id
        patient_data["systole"] = np.concatenate((patient_data["systole"], systole_prediction[None, :]), axis=0)
        patient_data["diastole"] = np.concatenate((patient_data["diastole"], diastole_prediction[None, :]), axis=0)

avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean=mean)
patient_targets = utils_heart.get_patient_average_heaviside_predictions(batch_targets, batch_ids, mean=mean)

assert avg_patient_predictions.viewkeys() == patient_targets.viewkeys()
crpss_sys, crpss_dst = [], []
for id in avg_patient_predictions.iterkeys():
    crpss_sys.append(utils_heart.crps(avg_patient_predictions[id][0], patient_targets[id][0]))
    crpss_dst.append(utils_heart.crps(avg_patient_predictions[id][1], patient_targets[id][1]))

crps0, crps1 = np.mean(crpss_sys), np.mean(crpss_dst)
print '\nValidation CRPS: ', crps0, crps1, 0.5 * (crps0 + crps1)
utils.save_pkl(avg_patient_predictions, valid_prediction_path)
print ' validation predictions saved to %s' % valid_prediction_path
print

# test
test_data_iterator = config().test_data_iterator
if n_tta_iterations == 1:
    test_data_iterator.transformation_params = config().valid_transformation_params
else:
    # just to be sure
    test_data_iterator.transformation_params['zoom_range'] = (1., 1.)

print 'test transformation params'
print test_data_iterator.transformation_params

print 'n test: %d' % test_data_iterator.nsamples

batch_predictions, batch_ids = [], []
for i in xrange(n_tta_iterations):
    print i,
    sys.stdout.flush()
    for xs_batch_valid, _, ids_batch in buffering.buffered_gen_threaded(test_data_iterator.generate()):
        for x_shared, x in zip(xs_shared, xs_batch_valid):
            x_shared.set_value(x)
        batch_predictions.append(iter_test_det())
        batch_ids.append(ids_batch)

for (systole_predictions, diastole_predictions), patient_ids in zip(batch_predictions, batch_ids):
    for systole_prediction, diastole_prediction, patient_id in zip(systole_predictions, diastole_predictions,
                                                                   patient_ids):
        patient_data = predictions[patient_id - 1]
        assert patient_data['patient'] == patient_id
        patient_data["systole"] = np.concatenate((patient_data["systole"], systole_prediction[None, :]), axis=0)
        patient_data["diastole"] = np.concatenate((patient_data["diastole"], diastole_prediction[None, :]), axis=0)

avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean=mean)

utils.save_pkl(avg_patient_predictions, test_prediction_path)
print '\npredictions saved to %s' % test_prediction_path

# utils.save_submission(avg_patient_predictions, submission_path)
# print ' submission saved to %s' % submission_path

try:
    with open(jonas_prediction_path, 'w') as f:
        pickle.dump({
            'metadata_path': metadata_path,
            'prediction_path': test_prediction_path,
            'submission_path': submission_path,
            'configuration_file': config().__name__,
            'git_revision_hash': utils.get_git_revision_hash(),
            'predictions': predictions
        }, f, pickle.HIGHEST_PROTOCOL)
except:
    with open('ira_%s.pkl' % config().__name__, 'w') as f:
        pickle.dump({
            'metadata_path': metadata_path,
            'prediction_path': test_prediction_path,
            'submission_path': submission_path,
            'configuration_file': config().__name__,
            'git_revision_hash': utils.get_git_revision_hash(),
            'predictions': predictions
        }, f, pickle.HIGHEST_PROTOCOL)
