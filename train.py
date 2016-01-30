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
import utils
from configuration import config, set_configuration

# import buffering

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")
config_name = sys.argv[1]
set_configuration(config_name)
expid = utils.generate_expid(config_name)
print
print "Experiment ID: %s" % expid
print

# metadata
if not os.path.isdir('metadata'):
    os.mkdir('metadata')
metadata_path = '/mnt/storage/metadata/kaggle-heart/train/%s.pkl' % expid

# logs
if not os.path.isdir('logs'):
    os.mkdir('logs')
# sys.stdout = open('/mnt/storage/metadata/kaggle-heart/logs/%s.log' % expid, 'w')  # use 2>&1 when running the script

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

train_loss = config().build_objective(model)

learning_rate_schedule = config().learning_rate_schedule
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
updates = config().build_updates(train_loss, model, learning_rate)

xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
ys_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_targets]

givens_in = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_in[l_in.input_var] = x

givens_out = {}
for l_target, y in izip(model.l_targets, ys_shared):
    givens_out[l_target.input_var] = y

givens = dict(givens_in.items() + givens_out.items())

# theano functions
iter_train = theano.function([], train_loss, givens=givens, updates=updates)
iter_validate = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_outs],
                                givens=givens_in)
test = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_params],
                                givens=givens_in)

if config().restart_from_save and os.path.isfile(metadata_path):
    print 'Load model parameters for resuming'
    resume_metadata = np.load(metadata_path)
    nn.layers.set_all_param_values(model.l_top, resume_metadata['param_values'])
    start_iter_idx = resume_metadata['iters_since_start'] + 1
    iter_idxs = range(start_iter_idx, config().max_niter)

    lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_iter_idx))
    print '  setting learning rate to %.7f' % lr
    learning_rate.set_value(lr)
    losses_train = resume_metadata['losses_train']
    losses_eval_valid = resume_metadata['losses_eval_valid']
else:
    iter_idxs = range(config().max_niter)
    losses_train = []
    losses_eval_valid = []

train_data_iterator = config().train_data_iterator
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples

print
print 'Train model'
start_time = time.time()
prev_time = start_time
tmp_losses_train = []

# buffering.buffered_gen_threaded(_gen())
for iter_idx, (xs_batch, ys_batch) in izip(iter_idxs, train_data_iterator.generate()):
    if iter_idx in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[iter_idx])
        print '  setting learning rate to %.7f' % lr
        print
        learning_rate.set_value(lr)

    prev_time = time.clock()
    # load data to GPU and make one iteration
    for x_shared, x in zip(xs_shared, xs_batch):
        x_shared.set_value(x)
    for y_shared, y in zip(ys_shared, ys_batch):
        y_shared.set_value(y)
    loss = iter_train()
    print loss, time.clock() - prev_time
    tmp_losses_train.append(loss)

    if ((iter_idx + 1) % config().validate_every) == 0:
        print
        print 'Iteration %d/%d' % (iter_idx + 1, config().max_niter)
        batch_valid_predictions, batch_valid_targets = [], []
        for xs_batch, ys_batch in valid_data_iterator.generate():
            for x_shared, x in zip(xs_shared, xs_batch):
                x_shared.set_value(x)
            batch_valid_targets.append(ys_batch)
            batch_valid_predictions.append(iter_validate())
            print '---------------------------------------'
            norm_params = test()
            print 'MU_0', norm_params[0]
            print 'sigma_0', norm_params[1]
            print 'MU_1', norm_params[2]
            print 'sigma_1', norm_params[3]

        # # calculate validation loss across validation set
        # valid_loss = config().get_mean_validation_loss(batch_valid_predictions, batch_valid_targets)
        # print 'Validation loss: ',  valid_loss
        #
        # valid_crps = config().get_mean_crps_loss(batch_valid_predictions, batch_valid_targets)
        # print 'Validation CRPS: ', valid_crps

        # calculate mean train loss since the last validation phase
        mean_train_loss = np.mean(tmp_losses_train)
        print 'Mean train loss: %7f' % mean_train_loss
        losses_train.append(mean_train_loss)
        tmp_losses_train = []

    if ((iter_idx + 1) % config().save_every) == 0:
        print
        print 'Saving metadata, parameters'

        with open(metadata_path, 'w') as f:
            pickle.dump({
                'configuration_file': config_name,
                'git_revision_hash': 0,  # utils.get_git_revision_hash(),
                'experiment_id': expid,
                'chunks_since_start': iter_idx,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                'param_values': nn.layers.get_all_param_values(model.l_top)
            }, f, pickle.HIGHEST_PROTOCOL)

        print '  saved to %s' % metadata_path
        print

    if iter_idx >= config().max_niter:
        break
