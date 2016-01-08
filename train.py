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
import theano.tensor as T
import time
from itertools import izip
from datetime import datetime, timedelta
import sys
from log import print_to_file
import logging
from functools import partial

def train_model(metadata_path, metadata=None):
    if metadata is None:
        if os.path.isfile(metadata_path):
            metadata = pickle.load(open(metadata_path, 'r'))

    print "Build model"
    interface_layers = config().build_model()

    output_layers = interface_layers["outputs"]
    input_layers = interface_layers["inputs"]
    top_layer = lasagne.layers.MergeLayer(
        incomings=output_layers.values()
    )
    all_layers = lasagne.layers.get_all_layers(top_layer)
    num_params = lasagne.layers.count_params(top_layer)
    print "  number of parameters: %d" % num_params
    print "  layer output shapes:"
    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 32)
        num_param = string.ljust(lasagne.layers.count_params(layer).__str__(), 10)
        print "    %s %s %s" % (name,  num_param, layer.output_shape)

    obj = config().build_objective(input_layers, output_layers)
    train_loss = obj.get_loss()
    kaggle_loss = obj.get_kaggle_loss()
    segmentation_loss = obj.get_segmentation_loss()

    outputs = [lasagne.layers.helper.get_output(output_layers[tag], deterministic=True) for tag in output_layers]

    all_params = lasagne.layers.get_all_params(top_layer)

    xs_shared = {
        key: lasagne.utils.shared_empty(dim=len(l_in.output_shape), dtype='float32') for (key, l_in) in input_layers.iteritems()
    }

    # contains target_vars of the objective! Not the output layers desired values!
    # There can be more output layers than are strictly required for the objective
    # e.g. for debugging
    ys_shared = {
        key: lasagne.utils.shared_empty(dim=target_var.ndim, dtype='float32') for (key, target_var) in obj.target_vars.iteritems()
    }

    learning_rate_schedule = config().learning_rate_schedule

    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
    idx = T.lscalar('idx')

    givens = dict()
    for key in output_layers.keys():
        if key in obj.target_vars:  #only add them when needed for our objective!
            givens[obj.target_vars[key]] = ys_shared[key][idx*config().batch_size : (idx+1)*config().batch_size]

    for key in input_layers.keys():
        givens[input_layers[key].input_var] = xs_shared[key][idx*config().batch_size:(idx+1)*config().batch_size]

    updates = config().build_updates(train_loss, all_params, learning_rate)
    grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(train_loss, all_params)]))

    iter_train = theano.function([idx], [train_loss, kaggle_loss, segmentation_loss], givens=givens, on_unused_input="ignore", updates=updates)
    compute_output = theano.function([idx], outputs, givens=givens, on_unused_input="ignore")

    if config().restart_from_save and os.path.isfile(metadata_path):
        print "Load model parameters for resuming"
        resume_metadata = np.load(metadata_path)
        lasagne.layers.set_all_param_values(top_layer, resume_metadata['param_values'])
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

    create_train_gen = partial(config().create_train_gen,
                               required_input_keys = xs_shared.keys(),
                               required_output_keys = ys_shared.keys()
                               )

    create_eval_valid_gen = partial(config().create_eval_valid_gen,
                               required_input_keys = xs_shared.keys(),
                               required_output_keys = ys_shared.keys()
                               )

    create_eval_train_gen = partial(config().create_eval_train_gen,
                               required_input_keys = xs_shared.keys(),
                               required_output_keys = ys_shared.keys()
                               )

    print "Train model"
    start_time = time.time()
    prev_time = start_time

    num_batches_chunk = config().batches_per_chunk

    for e, train_data in izip(chunks_train_idcs, create_train_gen()):
        print "Chunk %d/%d" % (e + 1, config().num_chunks_train)

        if e in learning_rate_schedule:
            lr = np.float32(learning_rate_schedule[e])
            print "  setting learning rate to %.7f" % lr
            learning_rate.set_value(lr)

        print "  load training data onto GPU"

        for key in xs_shared:
            xs_shared[key].set_value(train_data["input"][key])

        for key in ys_shared:
            ys_shared[key].set_value(train_data["output"][key])

        print "  batch SGD"
        losses = []
        kaggle_losses = []
        segmentation_losses = []
        for b in xrange(num_batches_chunk):
            loss, kaggle_loss, segmentation_loss = tuple(iter_train(b))
            losses.append(loss)
            kaggle_losses.append(kaggle_loss)
            segmentation_losses.append(segmentation_loss)

        mean_train_loss = np.mean(losses)
        print "  mean training loss:\t\t%.6f" % mean_train_loss
        losses_train.append(mean_train_loss)

        print "  mean kaggle loss:\t\t%.6f" % np.mean(kaggle_losses)
        print "  mean segment loss:\t\t%.6f" % np.mean(segmentation_losses)

        if ((e + 1) % config().validate_every) == 0:
            print
            print "Validating"
            subsets = ["train", "validation"]
            gens = [create_eval_train_gen, create_eval_valid_gen]
            label_sets = [config().get_label_set(subset)[1] for subset in subsets]
            losses_eval = [losses_eval_train, losses_eval_valid]

            for subset, create_gen, labels, losses in zip(subsets, gens, label_sets, losses_eval):
                print "  %s set (%d samples)" % (subset, len(labels))
                outputs = []
                for validation_data, chunk_length_eval in create_gen():
                    num_batches_chunk_eval = int(np.ceil(chunk_length_eval / float(config().batch_size)))

                    for key in xs_shared:
                        xs_shared[key].set_value(validation_data["input"][key])

                    outputs_chunk = []
                    for b in xrange(num_batches_chunk_eval):
                        out = compute_output(b) # TODO: this now returns all outputs as defined in the model!
                        out = config().postprocess(out)
                        outputs_chunk.append(out)

                    outputs_chunk = np.vstack(outputs_chunk)
                    outputs_chunk = outputs_chunk[:chunk_length_eval] # truncate to the right length
                    outputs.append(outputs_chunk)

                outputs = np.vstack(outputs)
                loss = utils.segmentation_log_loss(outputs, labels)
                acc = utils.segmentation_accuracy(outputs, labels)
                utils.segmentation_visualization(outputs, labels)
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

    expid = utils.generate_expid(args.config)

    with print_to_file("/mnt/storage/metadata/kaggle-heart/logs/%s.log" % expid):

        print "Running configuration:", config().__name__
        print "Current git version:", utils.get_git_revision_hash()
        metadata_path = "/mnt/storage/metadata/kaggle-heart/train/%s.pkl" % expid

        meta_data = train_model(metadata_path)
        predict_model(metadata_path)


