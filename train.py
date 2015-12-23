from __future__ import division
import argparse
from configuration import config, set_configuration
import numpy as np
import string
from predict import predict_model
import utils
import cPickle as pickle
import os
import lasagne
import theano
import theano.Tensor as T
import time
from itertools import izip
from datetime import datetime, timedelta


def train_model(metadata_path, metadata=None):
    if metadata is None:
        if os.path.isfile(metadata_path):
            metadata = pickle.load(open(open(metadata_path, 'r')))

    print "Build model"
    interface_layers = config().build_model()

    output_layer = interface_layers["output"]
    input_layers = interface_layers["inputs"]
    all_layers = lasagne.layers.get_all_layers(output_layer)
    num_params = lasagne.layers.count_params(output_layer)
    print "  number of parameters: %d" % num_params
    print "  layer output shapes:"
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        print "    %s %s" % (name, layer.get_output_shape(),)

    obj = config().build_objective(input_layers, output_layer)
    train_loss = obj.get_loss()
    output = output_layer.get_output(deterministic=True)

    all_params = lasagne.layers.get_all_params(output_layer)

    input_ndims = [len(l_in.get_output_shape()) for l_in in input_layers]
    xs_shared = [lasagne.utils.shared_empty(dim=ndim) for ndim in input_ndims]
    y_shared = lasagne.utils.shared_empty(dim=3)

    learning_rate_schedule = config().learning_rate_schedule

    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

    idx = T.lscalar('idx')

    givens = {
        obj.target_var: y_shared[idx*config().batch_size:(idx+1)*config().batch_size],
    }

    for l_in, x_shared in zip(input_layers, xs_shared):
         givens[l_in.input_var] = x_shared[idx*config().batch_size:(idx+1)*config().batch_size]

    updates = config().build_updates(train_loss, all_params, learning_rate)

    iter_train = theano.function([idx], train_loss, givens=givens, updates=updates)
    compute_output = theano.function([idx], output, givens=givens, on_unused_input="ignore")

    if config().restart_from_save:
        print "Load model parameters for resuming"
        if os.file.exists(metadata_path):
            resume_metadata = np.load(metadata_path)
            lasagne.layers.set_all_param_values(output_layer, resume_metadata['param_values'])
            start_chunk_idx = resume_metadata['chunks_since_start'] + 1
            chunks_train_idcs = range(start_chunk_idx, config().num_chunks_train)

            # set lr to the correct value
            current_lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
            print "  setting learning rate to %.7f" % current_lr
            learning_rate.set_value(current_lr)
            losses_train = resume_metadata['losses_train']
            losses_eval_valid = resume_metadata['losses_eval_valid']
            losses_eval_train = resume_metadata['losses_eval_train']
        else:
            chunks_train_idcs = range(config().num_chunks_train)
            losses_train = []
            losses_eval_valid = []
            losses_eval_train = []


    create_train_gen = config().create_train_gen
    create_eval_valid_gen = config().create_eval_valid_gen
    create_eval_train_gen = config().create_eval_train_gen

    print "Train model"
    start_time = time.time()
    prev_time = start_time

    num_batches_chunk = config().chunk_size // config().batch_size


    for e, (xs_chunk, y_chunk) in izip(chunks_train_idcs, create_train_gen()):
        print "Chunk %d/%d" % (e + 1, config().num_chunks_train)

        if e in learning_rate_schedule:
            lr = np.float32(learning_rate_schedule[e])
            print "  setting learning rate to %.7f" % lr
            learning_rate.set_value(lr)

        print "  load training data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)
        y_shared.set_value(y_chunk)

        print "  batch SGD"
        losses = []
        for b in xrange(num_batches_chunk):
            loss = iter_train(b)
            losses.append(loss)


        mean_train_loss = np.mean(losses)
        print "  mean training loss:\t\t%.6f" % mean_train_loss
        losses_train.append(mean_train_loss)

        if ((e + 1) % config().validate_every) == 0:
            print
            print "Validating"
            subsets = ["train", "valid"]
            gens = [create_eval_train_gen, create_eval_valid_gen]
            label_sets = [config().data_loader.labels_train, config().data_loader.labels_valid]
            losses_eval = [losses_eval_train, losses_eval_valid]

            for subset, create_gen, labels, losses in zip(subsets, gens, label_sets, losses_eval):
                print "  %s set" % subset
                outputs = []
                for xs_chunk_eval, chunk_length_eval in create_gen():
                    num_batches_chunk_eval = int(np.ceil(chunk_length_eval / float(config().batch_size)))

                    for x_shared, x_chunk_eval in zip(xs_shared, xs_chunk_eval):
                        x_shared.set_value(x_chunk_eval)

                    outputs_chunk = []
                    for b in xrange(num_batches_chunk_eval):
                        out = compute_output(b)
                        outputs_chunk.append(out)

                    outputs_chunk = np.vstack(outputs_chunk)
                    outputs_chunk = outputs_chunk[:chunk_length_eval] # truncate to the right length
                    outputs.append(outputs_chunk)

                outputs = np.vstack(outputs)
                loss = utils.log_loss(outputs, labels)
                acc = utils.accuracy(outputs, labels)
                print "    loss:\t%.6f" % loss
                print "    acc:\t%.2f%%" % (acc * 100)
                print

                losses.append(loss)
                del outputs


        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (float(config().num_chunks_train - (e + 1)) / float(e + 1 - chunks_train_idcs[0]))
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
        print

        if ((e + 1) % config().save_every) == 0:
            print
            print "Saving metadata, parameters"

            with open(metadata_path, 'w') as f:
                pickle.dump({
                    'metadata_path': metadata_path,
                    'configuration_file': config().__name__,
                    'git_revision_hash': utils.get_git_revision_hash(),
                    'experiment_id': expid,
                    'chunks_since_start': e,
                    'losses_train': losses_train,
                    'losses_eval_valid': losses_eval_valid,
                    'losses_eval_train': losses_eval_train,
                    'time_since_start': time_since_start,
                    'param_values': lasagne.layers.get_all_param_values(output_layer)
                }, f, pickle.HIGHEST_PROTOCOL)

            print "  saved to %s" % metadata_path
            print

    return metadata



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    args = parser.parse_args()
    set_configuration(args.config)
    print config().__name__
    print utils.get_git_revision_hash()
    expid = utils.generate_expid(args.config)
    metadata_path = "metadata/%s.pkl" % expid

    meta_data = train_model(metadata_path)
    predict_model(metadata_path)


