import sys
import numpy as np
import theano
from itertools import izip
import lasagne as nn
import utils
import buffering
import utils_heart
from configuration import config, set_configuration, set_subconfiguration
import pathfinder

if not (3 <= len(sys.argv) <= 5):
    sys.exit("Usage: predict.py <config_name> <set: train|valid|test> <n_tta_iterations> "
             "<average: arithmetic|geometric>")

config_name = sys.argv[1]
set = sys.argv[2] if len(sys.argv) >= 3 else 'valid'
n_tta_iterations = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
mean = sys.argv[4] if len(sys.argv) >= 5 else 'geometric'

print 'Make %s tta predictions for %s set using %s mean' % (n_tta_iterations, set, mean)

metadata_dir = utils.get_dir_path('train', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)
metadata = utils.load_pkl(metadata_path)

assert config_name == metadata['configuration']
if 'subconfiguration' in metadata:
    set_subconfiguration(metadata['subconfiguration'])
set_configuration(config_name)

# predictions paths
prediction_dir = utils.get_dir_path('predictions', pathfinder.METADATA_PATH)
prediction_path = prediction_dir + "/%s-%s-%s-%s.pkl" % (metadata['experiment_id'], set, n_tta_iterations, mean)

# submissions paths
submission_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
submission_path = submission_dir + "/%s-%s-%s-%s.csv" % (metadata['experiment_id'], set, n_tta_iterations, mean)

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

if set == 'train':
    train_data_iterator = config().train_data_iterator
    if n_tta_iterations == 1:
        train_data_iterator.transformation_params = config().valid_transformation_params
    else:
        train_data_iterator.transformation_params = config().train_transformation_params

    train_data_iterator.transformation_params['zoom_range'] = (1., 1.)
    train_data_iterator.full_batch = False
    train_data_iterator.random = False
    train_data_iterator.infinite = False

    print
    print 'n train: %d' % train_data_iterator.nsamples
    print 'tta iteration:',

    batch_predictions, batch_targets, batch_ids = [], [], []
    for i in xrange(n_tta_iterations):
        print i,
        sys.stdout.flush()
        for xs_batch_valid, ys_batch_valid, ids_batch in buffering.buffered_gen_threaded(
                train_data_iterator.generate()):
            for x_shared, x in zip(xs_shared, xs_batch_valid):
                x_shared.set_value(x)

            batch_targets.append(ys_batch_valid)
            batch_predictions.append(iter_test_det())
            batch_ids.append(ids_batch)

    avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean=mean)
    patient_targets = utils_heart.get_patient_average_heaviside_predictions(batch_targets, batch_ids)

    assert avg_patient_predictions.viewkeys() == patient_targets.viewkeys()
    crpss_sys, crpss_dst = [], []
    for id in avg_patient_predictions.iterkeys():
        crpss_sys.append(utils_heart.crps(avg_patient_predictions[id][0], patient_targets[id][0]))
        crpss_dst.append(utils_heart.crps(avg_patient_predictions[id][1], patient_targets[id][1]))
        print id, 0.5 * (crpss_sys[-1] + crpss_dst[-1]), crpss_sys[-1], crpss_dst[-1]

    crps0, crps1 = np.mean(crpss_sys), np.mean(crpss_dst)

    print '\n Train CRPS:', config().get_mean_crps_loss(batch_predictions, batch_targets, batch_ids)
    print 'Train CRPS', 0.5 * (crps0 + crps1)

    utils.save_pkl(avg_patient_predictions, prediction_path)
    print ' predictions saved to %s' % prediction_path
    print

if set == 'valid':
    valid_data_iterator = config().valid_data_iterator
    if n_tta_iterations > 1:
        valid_data_iterator.transformation_params = config().train_transformation_params
        valid_data_iterator.transformation_params['zoom_range'] = (1., 1.)

    print
    print 'n valid: %d' % valid_data_iterator.nsamples
    print 'tta iteration:',

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

    avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean=mean)
    patient_targets = utils_heart.get_patient_average_heaviside_predictions(batch_targets, batch_ids)

    assert avg_patient_predictions.viewkeys() == patient_targets.viewkeys()
    crpss_sys, crpss_dst = [], []
    for id in avg_patient_predictions.iterkeys():
        crpss_sys.append(utils_heart.crps(avg_patient_predictions[id][0], patient_targets[id][0]))
        crpss_dst.append(utils_heart.crps(avg_patient_predictions[id][1], patient_targets[id][1]))
        print id, 0.5 * (crpss_sys[-1] + crpss_dst[-1]), crpss_sys[-1], crpss_dst[-1]

    crps0, crps1 = np.mean(crpss_sys), np.mean(crpss_dst)

    print '\n Validation CRPS:', config().get_mean_crps_loss(batch_predictions, batch_targets, batch_ids)
    print 'Validation CRPS: ', crps0, crps1, 0.5 * (crps0 + crps1)

    utils.save_pkl(avg_patient_predictions, prediction_path)
    print ' predictions saved to %s' % prediction_path
    print

if set == 'test':
    test_data_iterator = config().test_data_iterator

    if n_tta_iterations == 1:
        test_data_iterator.transformation_params = config().valid_transformation_params
    else:
        test_data_iterator.transformation_params['zoom_range'] = (1., 1.)

    print 'n test: %d' % test_data_iterator.nsamples
    print 'tta iteration:',

    batch_predictions, batch_ids = [], []
    for i in xrange(n_tta_iterations):
        print i,
        sys.stdout.flush()
        for xs_batch_test, _, ids_batch in buffering.buffered_gen_threaded(test_data_iterator.generate()):
            for x_shared, x in zip(xs_shared, xs_batch_test):
                x_shared.set_value(x)
            batch_predictions.append(iter_test_det())
            batch_ids.append(ids_batch)

    avg_patient_predictions = config().get_avg_patient_predictions(batch_predictions, batch_ids, mean=mean)
    utils.save_pkl(avg_patient_predictions, prediction_path)
    print ' predictions saved to %s' % prediction_path

    utils.save_submission(avg_patient_predictions, submission_path)
    print ' submission saved to %s' % submission_path
