import cPickle as pickle
import os
import string
import sys
import time
from itertools import izip

import lasagne as nn
import numpy as np
import theano
import theano.tensor as T

import buffering
import utils
from configuration import config, set_configuration

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")
config_name = sys.argv[1]

set_configuration(config)
expid = utils.generate_expid(config_name)
print
print "Experiment ID: %s" % expid
print

metadata_path = '/mnt/storage/metadata/kaggle-heart/train/%s.pkl' % expid
sys.stdout = open('/mnt/storage/metadata/kaggle-heart/logs/%s.log' % expid, 'w')  # use 2>&1 when running the script

print 'Build model'
model = config().build_model()
all_layers = nn.layers.get_all_layers(model.top_layer)
all_params = nn.layers.get_all_params(model.top_layer)
num_params = nn.layers.count_params(model.top_layer)
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers[:-1]:
    name = string.ljust(layer.__class__.__name__, 32)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(num_param.__str__(), 10)
    print '    %s %s %s' % (name, num_param, layer.output_shape)

train_loss = config().build_objective(model)
valid_loss = config().build_objective(model, deterministic=True)

learning_rate_schedule = config().learning_rate_schedule
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
updates = config().build_updates(train_loss, model, learning_rate)

idx = T.lscalar('idx')
xs_shared = [nn.utils.shared_empty(dim=l.shape) for l in model.l_ins]
ys_shared = [nn.utils.shared_empty(dim=l.shape) for l in model.l_targets]

givens = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens[l_in.input_var] = x

for l_target, y in izip(model.l_targets, ys_shared):
    givens[l_target.input_var] = y

iter_train = theano.function([], train_loss,
                             givens=givens, updates=updates)
iter_validate = theano.function([], valid_loss,
                                givens=givens)

if config().restart_from_save and os.path.isfile(metadata_path):
    print 'Load model parameters for resuming'
    resume_metadata = np.load(metadata_path)
    nn.layers.set_all_param_values(model.top_layer, resume_metadata['param_values'])
    start_chunk_idx = resume_metadata['chunks_since_start'] + 1
    chunks_train_idcs = range(start_chunk_idx, config().num_chunks_train)

    # set lr to the correct value
    current_lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
    print '  setting learning rate to %.7f' % current_lr
    learning_rate.set_value(current_lr)
    losses_train = resume_metadata['losses_train']
    losses_eval_valid = resume_metadata['losses_eval_valid']
else:
    chunks_train_idcs = range(config().num_chunks_train)
    losses_train = []
    losses_eval_valid = []


print "Load data"
config().data_iterator.load_train()
if config().create_train_gen:
    create_train_gen = config().create_train_gen
else:
    create_train_gen = lambda: config().data_iterator.create_random_gen(config().data_iterator.images_train,
                                                                        config().data_iterator.labels_train)

if config().create_eval_valid_gen:
    create_eval_valid_gen = config().create_eval_valid_gen
else:
    create_eval_valid_gen = lambda: config().data_iterator.create_fixed_gen(config().data_iterator.images_valid,
                                                                            augment=False)

print 'Train model'
start_time = time.time()
prev_time = start_time

for e, (xs_batch, ys_batch) in izip(chunks_train_idcs, buffering.buffered_gen_threaded(create_train_gen())):
    print 'Batch %d/%d' % (e + 1, config().num_batchess_train)

    if e in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[e])
        print '  setting learning rate to %.7f' % lr
        learning_rate.set_value(lr)

    # load data to GPU and make one iteration
    for x_shared, x in zip(xs_shared, xs_batch):
        x_shared.set_value(x)
    for y_shared, y in zip(ys_shared, ys_batch):
        y_shared.set_value(y)
    loss = iter_train()

    if ((e + 1) % config().validate_every) == 0:
        pass

    if ((e + 1) % config().save_every) == 0:
        print
        print 'Saving metadata, parameters'

        with open(metadata_path, 'w') as f:
            pickle.dump({
                'configuration_file': config_name,
                'git_revision_hash': utils.get_git_revision_hash(),
                'experiment_id': expid,
                'chunks_since_start': e,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                'param_values': nn.layers.get_all_param_values(model.top_layer)
            }, f, pickle.HIGHEST_PROTOCOL)

        print '  saved to %s' % metadata_path
        print
