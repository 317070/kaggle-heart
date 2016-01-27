import argparse
from configuration import config, set_configuration
import numpy as np
import string
import utils
import cPickle as pickle
import os
import lasagne as nn
import theano
import theano.tensor as T
import time
from itertools import izip
import sys
from functools import partial
import buffering

parser = argparse.ArgumentParser(description=__doc__)
required = parser.add_argument_group('required arguments')
required.add_argument('-c', '--config',
                      help='configuration to run',
                      required=True)
args = parser.parse_args()
set_configuration(args.config)

expid = utils.generate_expid(args.config)

print 'Running configuration:', config().__name__
print 'Current git version:', utils.get_git_revision_hash()
metadata_path = '/mnt/storage/metadata/kaggle-heart/train/%s.pkl' % expid
sys.stdout = open('/mnt/storage/metadata/kaggle-heart/logs/%s.log' % expid, 'w')  # use 2>&1 when running the script

print 'Build model'
model = config().build_model()

output_layers = model.l_outs
input_layers = model.l_ins
top_layer = nn.layers.MergeLayer(incomings=output_layers)
all_layers = nn.layers.get_all_layers(top_layer)
all_params = nn.layers.get_all_params(top_layer)
num_params = nn.layers.count_params(top_layer)
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers[:-1]:
    name = string.ljust(layer.__class__.__name__, 32)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(num_param.__str__(), 10)
    print '    %s %s %s' % (name, num_param, layer.output_shape)

predictions = [nn.layers.get_output(l) for l in output_layers]
targets = [T.fmatrix() for l in output_layers]
train_loss = config().build_objective(predictions, targets)

predictions_det = [nn.layers.get_output(l, deterministic=True) for l in output_layers]
valid_loss = config().build_objective(predictions_det, targets)

learning_rate_schedule = config().learning_rate_schedule
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

if hasattr(config(), 'build_updates'):
    updates = config().build_updates(train_loss.loss, all_params, learning_rate)
else:
    updates = nn.updates.adam(train_loss.loss, all_params, learning_rate)

idx = T.lscalar('idx')
givens = {}

iter_train = theano.function([idx], [train_loss.loss],
                             givens=givens, updates=updates)
iter_validate = theano.function([idx], [valid_loss.loss],
                                givens=givens)

if config().restart_from_save and os.path.isfile(metadata_path):
    print 'Load model parameters for resuming'
    resume_metadata = np.load(metadata_path)
    nn.layers.set_all_param_values(top_layer, resume_metadata['param_values'])
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
    losses_eval_train = []

create_train_gen = partial(config().create_train_gen,
                           required_input_keys=xs_shared.keys(),
                           required_output_keys=ys_shared.keys(),
                           )

create_eval_valid_gen = partial(config().create_eval_valid_gen,
                                required_input_keys=xs_shared.keys(),
                                required_output_keys=ys_shared.keys()
                                )

create_eval_train_gen = partial(config().create_eval_train_gen,
                                required_input_keys=xs_shared.keys(),
                                required_output_keys=ys_shared.keys()
                                )

print 'Train model'
start_time = time.time()
prev_time = start_time

num_batches_chunk = config().batches_per_chunk

for e, train_data in izip(chunks_train_idcs, buffering.buffered_gen_threaded(create_train_gen())):
    print 'Chunk %d/%d' % (e + 1, config().num_chunks_train)

    if e in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[e])
        print '  setting learning rate to %.7f' % lr
        learning_rate.set_value(lr)

    print '  load training data onto GPU'
    for key in xs_shared:
        xs_shared[key].set_value(train_data['input'][key])

    for key in ys_shared:
        ys_shared[key].set_value(train_data['output'][key])

    print '  batch SGD'
    losses = []
    kaggle_losses = []
    segmentation_losses = []
    for b in xrange(num_batches_chunk):
        iter_result = iter_train(b)

        loss, kaggle_loss, segmentation_loss = tuple(iter_result[:3])

        losses.append(loss)
        kaggle_losses.append(kaggle_loss)
        segmentation_losses.append(segmentation_loss)

    mean_train_loss = np.mean(losses)
    print '  mean training loss:\t\t%.6f' % mean_train_loss
    losses_train.append(mean_train_loss)

    print '  mean kaggle loss:\t\t%.6f' % np.mean(kaggle_losses)
    print '  mean segment loss:\t\t%.6f' % np.mean(segmentation_losses)

    if ((e + 1) % config().validate_every) == 0:

        pass

    if ((e + 1) % config().save_every) == 0:
        print
        print 'Saving metadata, parameters'

        with open(metadata_path, 'w') as f:
            pickle.dump({
                'configuration_file': config().__name__,
                'git_revision_hash': utils.get_git_revision_hash(),
                'experiment_id': expid,
                'chunks_since_start': e,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                'param_values': nn.layers.get_all_param_values(top_layer)
            }, f, pickle.HIGHEST_PROTOCOL)

        print '  saved to %s' % metadata_path
        print
