import cPickle as pickle
import string
import sys
import time
from itertools import izip
import lasagne as nn
import numpy as np
import theano
from datetime import datetime, timedelta
import utils
import logger
import theano.tensor as T
import buffering
from configuration import config, set_configuration, set_subconfiguration
import pathfinder

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <meta_configuration_name>")

config_name = sys.argv[1]

subconfig_name = config_name.replace('meta_', '')
metadata_dir = utils.get_dir_path('train', pathfinder.MODEL_PATH)
submodel_metadata_path = utils.find_model_metadata(metadata_dir, subconfig_name)
submodel_metadata = utils.load_pkl(submodel_metadata_path)

assert subconfig_name == submodel_metadata['configuration']
set_subconfiguration(subconfig_name)
set_configuration(config_name)

expid = utils.generate_expid(config_name)
print
print "Experiment ID: %s" % expid
print

# meta metadata and logs paths
metadata_path = metadata_dir + '/%s.pkl' % expid
logs_dir = utils.get_dir_path('logs', pathfinder.MODEL_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % expid)

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

nn.layers.set_all_param_values(model.submodel.l_top, submodel_metadata['param_values'])

train_loss = config().build_objective(model)

learning_rate_schedule = config().learning_rate_schedule
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
updates = config().build_updates(train_loss, model, learning_rate)

xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
ys_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_targets]

idx = T.lscalar('idx')
givens_train = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_train[l_in.input_var] = x[idx * config().batch_size:(idx + 1) * config().batch_size]
for l_target, y in izip(model.l_targets, ys_shared):
    givens_train[l_target.input_var] = y[idx * config().batch_size:(idx + 1) * config().batch_size]

givens_valid = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_valid[l_in.input_var] = x

# theano functions
iter_train = theano.function([idx], train_loss, givens=givens_train, updates=updates, on_unused_input='ignore')
iter_validate = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_outs],
                                givens=givens_valid, on_unused_input='ignore')

if config().restart_from_save:
    print 'Load model parameters for resuming'
    resume_metadata = utils.load_pkl(config().restart_from_save)
    nn.layers.set_all_param_values(model.l_top, resume_metadata['param_values'])
    start_chunk_idx = resume_metadata['chunks_since_start'] + 1
    chunk_idxs = range(start_chunk_idx, config().max_nchunks)

    lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
    print '  setting learning rate to %.7f' % lr
    learning_rate.set_value(lr)
    losses_train = resume_metadata['losses_train']
    losses_eval_valid = resume_metadata['losses_eval_valid']
    crps_eval_valid = resume_metadata['crps_eval_valid']
else:
    chunk_idxs = range(config().max_nchunks)
    losses_train = []
    losses_eval_valid = []
    crps_eval_valid = []
    start_chunk_idx = 0

train_data_iterator = config().train_data_iterator
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples

print
print 'Train model'
chunk_idx = 0
start_time = time.time()
prev_time = start_time
tmp_losses_train = []

for chunk_idx, (xs_chunk, ys_chunk, patient_idx) in izip(chunk_idxs,
                                                         buffering.buffered_gen_threaded(
                                                             train_data_iterator.generate())):
    if chunk_idx in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[chunk_idx])
        print '  setting learning rate to %.7f' % lr
        print
        learning_rate.set_value(lr)

    # load chunk to GPU
    for x_shared, x in zip(xs_shared, xs_chunk):
        x_shared.set_value(x)
    for y_shared, y in zip(ys_shared, ys_chunk):
        y_shared.set_value(y)

    # make nbatches_chunk iterations
    for b in xrange(config().nbatches_chunk):
        loss = iter_train(b)
        tmp_losses_train.append(loss)

    if ((chunk_idx + 1) % config().validate_every) == 0:
        print
        print 'Chunk %d/%d' % (chunk_idx + 1, config().max_nchunks)
        # calculate mean train loss since the last validation phase
        mean_train_loss = np.mean(tmp_losses_train)
        print 'Mean train loss: %7f' % mean_train_loss
        losses_train.append(mean_train_loss)
        tmp_losses_train = []

        # load validation data to GPU
        batch_valid_predictions, batch_valid_targets, batch_valid_ids = [], [], []
        for xs_batch_valid, ys_batch_valid, ids_batch in valid_data_iterator.generate():
            for x_shared, x in zip(xs_shared, xs_batch_valid):
                x_shared.set_value(x)

            batch_valid_targets.append(ys_batch_valid)
            batch_valid_predictions.append(iter_validate())
            batch_valid_ids.append(ids_batch)

        # calculate validation loss across validation set
        valid_crps = config().get_mean_crps_loss(batch_valid_predictions, batch_valid_targets, batch_valid_ids)
        print 'Validation CRPS: ', valid_crps, np.mean(valid_crps)
        crps_eval_valid.append(valid_crps)

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (config().max_nchunks - chunk_idx + 1.) / (chunk_idx + 1. - start_chunk_idx)
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
        print

    if ((chunk_idx + 1) % config().save_every) == 0:
        print
        print 'Saving metadata, parameters'

        with open(metadata_path, 'w') as f:
            pickle.dump({
                'configuration': config_name,
                'subconfiguration': subconfig_name,
                'git_revision_hash': utils.get_git_revision_hash(),
                'experiment_id': expid,
                'chunks_since_start': chunk_idx,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                'crps_eval_valid': crps_eval_valid,
                'param_values': nn.layers.get_all_param_values(model.l_top)
            }, f, pickle.HIGHEST_PROTOCOL)

            print '  saved to %s' % metadata_path
            print
