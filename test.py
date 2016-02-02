import sys
import numpy as np
import theano
import theano.tensor as T
from itertools import izip
import lasagne as nn
import os
import string
import utils
import buffering
from configuration import config, set_configuration

if not (3 <= len(sys.argv) <= 5):
    sys.exit("Usage: test.py <metadata_path> <test_method>")

metadata_path = sys.argv[1]
method = sys.argv[2]

# TODO assert method

print "Load parameters"
metadata = np.load('/mnt/storage/metadata/kaggle-heart/train/ira/' + metadata_path)

config_name = metadata['configuration']
set_configuration(config_name)

# predictions
prediction_dir = '/mnt/storage/metadata/kaggle-heart/predictions/ira'
if not os.path.isdir(prediction_dir):
    os.mkdir(prediction_dir)
predictions_path = prediction_dir + "/%s--%s.npy" % (metadata['experiment_id'], method)

print "Build model"
model = config().build_model()
all_layers = nn.layers.get_all_layers(model.l_top)
all_params = nn.layers.get_all_params(model.l_top)
num_params = nn.layers.count_params(model.l_top)
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers[:-1]:
    name = string.ljust(layer.__class__.__name__, 32)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(num_param.__str__(), 10)
    print '    %s %s %s' % (name, num_param, layer.output_shape)

nn.layers.set_all_param_values(model.l_top, metadata['param_values'])

valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n test: %d' % valid_data_iterator.nsamples

xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
givens_in = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_in[l_in.input_var] = x

iter_test_det = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_outs],
                                givens=givens_in)
iter_test_nodet = theano.function([], [nn.layers.get_output(l, deterministic=False) for l in model.l_outs],
                                  givens=givens_in)

# validation set predictions
batch_predictions, batch_targets, batch_ids = [], [], []
for _ in xrange(100):
    for xs_batch_valid, ys_batch_valid, ids_batch in buffering.buffered_gen_threaded(valid_data_iterator.generate()):
        for x_shared, x in zip(xs_shared, xs_batch_valid):
            x_shared.set_value(x)
        batch_targets.append(ys_batch_valid)
        batch_predictions.append(iter_test_det())
        batch_ids.append(ids_batch)

avg_patient_predictions = utils.get_avg_patient_predictions(batch_predictions, batch_ids)
patient_targets = utils.get_avg_patient_predictions(batch_targets, batch_ids)

assert avg_patient_predictions.viewkeys() == patient_targets.viewkeys()
crpss_sys, crpss_dst = [], []
for id in avg_patient_predictions.iterkeys():
    crpss_sys.append(utils.crps(avg_patient_predictions[id][0], patient_targets[id][0]))
    crpss_dst.append(utils.crps(avg_patient_predictions[id][1], patient_targets[id][1]))

print 'Validation Systole CRPS: ', np.mean(crpss_sys)
print 'Validation Diastole CRPS: ', np.mean(crpss_dst)

test_data_iterator = config().test_data_iterator
print
print 'Data'
print 'n test: %d' % valid_data_iterator.nsamples
