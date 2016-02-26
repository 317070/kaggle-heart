import string
import sys
from itertools import izip
import lasagne as nn
import numpy as np
import theano
import utils
import theano.tensor as T
from configuration import config, set_configuration
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit("Usage: dump_train.py <model_metadata>")

metadata_dir = utils.get_dir_path('train')
model_metadata = utils.load_pkl(metadata_dir + '/%s' % sys.argv[1])
set_configuration(model_metadata['configuration'])

print 'Build model'
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

nn.layers.set_all_param_values(model.l_top, model_metadata['param_values'])

xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
ys_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_targets]

idx = T.lscalar('idx')
givens_train = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_train[l_in.input_var] = x
for l_target, y in izip(model.l_targets, ys_shared):
    givens_train[l_target.input_var] = y

# theano functions
get_predictions = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_outs],
                                  givens=givens_train, on_unused_input='ignore')
get_targets = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_targets],
                              givens=givens_train, on_unused_input='ignore')

train_data_iterator = config().train_data_iterator
valid_data_iterator = config().valid_data_iterator

# no augmentation
train_data_iterator.transformation_params = valid_data_iterator.transformation_params
train_data_iterator.random = False
train_data_iterator.full_batch = False
train_data_iterator.infinite = False

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples

print
print 'Train model'
chunk_idx = 0
predictions0, predictions1 = [], []
targets0, targets1 = [], []
patients_ids = []

chunk_idxs = range(config().nchunks_per_epoch)
print chunk_idxs[-1]

for (xs_chunk, ys_chunk, patient_batch_ids) in train_data_iterator.generate():

    # load chunk to GPU
    for x_shared, x in zip(xs_shared, xs_chunk):
        x_shared.set_value(x)
    for y_shared, y in zip(ys_shared, ys_chunk):
        y_shared.set_value(y)

    p = get_predictions()
    predictions0.append(p[0])
    predictions1.append(p[1])

    t = get_targets()
    print t
    targets0.append(t[0])
    targets1.append(t[1])

    patients_ids.extend(patient_batch_ids)

predictions0 = np.vstack(predictions0)
predictions1 = np.vstack(predictions1)

targets0 = np.vstack(targets0)
targets1 = np.vstack(targets1)

patient2idxs = defaultdict(list)
for i, pid in enumerate(patients_ids):
    patient2idxs[pid].append(i)
print patient2idxs.keys()
print len(patient2idxs.keys())

patient2predictions0 = {}
patient2predictions1 = {}
patient2targets0 = {}
patient2targets1 = {}

for pid, idxs in patient2idxs.iteritems():
    patient2predictions0[pid] = predictions0[idxs]
    patient2predictions1[pid] = predictions1[idxs]
    patient2targets0[pid] = targets0[idxs]
    patient2targets1[pid] = targets1[idxs]

d = {'p0': patient2predictions0, 'p1': patient2predictions1, 't0': patient2targets0, 't1': patient2targets1}
utils.save_pkl(d, '/home/ikorshun/kaggle-heart/train.pkl')

print patient2predictions0[430]
print patient2predictions1[430]
print patient2targets0[430]
print patient2targets1[430]
print '-----------------------------------'
