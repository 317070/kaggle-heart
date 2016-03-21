"""Script for training a network as defined in a given configuraton.

Usage:
> python train.py -c CONFIG_NAME
"""

from __future__ import division

import argparse
import collections
import cPickle as pickle
import logging
import string
import time
import os
import sys

from datetime import datetime, timedelta
from functools import partial
from itertools import izip

import lasagne
import numpy as np
import theano
import theano.tensor as T

from theano.compile.nanguardmode import NanGuardMode

import buffering
import data_loader
import theano_printer
import utils

from configuration import config, set_configuration
from data_loader import get_lenght_of_set, get_number_of_validation_samples, validation_patients_indices, NUM_TRAIN_PATIENTS
from paths import MODEL_PATH, LOGS_PATH
from predict import predict_model
from log import print_to_file


def train_model(expid):
    metadata_path = MODEL_PATH + "%s.pkl" % expid

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    data_loader.filter_patient_folders()

    print "Build model"
    interface_layers = config().build_model()

    output_layers = interface_layers["outputs"]
    input_layers = interface_layers["inputs"]
    top_layer = lasagne.layers.MergeLayer(
        incomings=output_layers.values()
    )
    all_layers = lasagne.layers.get_all_layers(top_layer)

    all_params = lasagne.layers.get_all_params(top_layer, trainable=True)
    if "cutoff_gradients" in interface_layers:
        submodel_params = [param for value in interface_layers["cutoff_gradients"] for param in lasagne.layers.get_all_params(value)]
        all_params = [p for p in all_params if p not in submodel_params]

    if "pretrained" in interface_layers:
        for config_name, layers_dict in interface_layers["pretrained"].iteritems():
            pretrained_metadata_path = MODEL_PATH + "%s.pkl" % config_name.split('.')[1]
            pretrained_resume_metadata = np.load(pretrained_metadata_path)
            pretrained_top_layer = lasagne.layers.MergeLayer(
                incomings = layers_dict.values()
            )
            lasagne.layers.set_all_param_values(pretrained_top_layer, pretrained_resume_metadata['param_values'])

    num_params = sum([np.prod(p.get_value().shape) for p in all_params])

    print string.ljust("  layer output shapes:",36),
    print string.ljust("#params:",10),
    print string.ljust("#data:",10),
    print "output shape:"
    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 32)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(int(num_param).__str__(), 10)
        num_size = string.ljust(np.prod(layer.output_shape[1:]).__str__(), 10)
        print "    %s %s %s %s" % (name,  num_param, num_size, layer.output_shape)
    print "  number of parameters: %d" % num_params

    obj = config().build_objective(interface_layers)

    train_loss_theano = obj.get_loss()
    kaggle_loss_theano = obj.get_kaggle_loss()
    segmentation_loss_theano = obj.get_segmentation_loss()

    validation_other_losses = collections.OrderedDict()
    validation_train_loss = obj.get_loss(average=False, deterministic=True, validation=True, other_losses=validation_other_losses)
    validation_kaggle_loss = obj.get_kaggle_loss(average=False, deterministic=True, validation=True)
    validation_segmentation_loss = obj.get_segmentation_loss(average=False, deterministic=True, validation=True)


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
    for key in obj.target_vars.keys():
        if key=="segmentation":
            givens[obj.target_vars[key]] = ys_shared[key][idx*config().sunny_batch_size : (idx+1)*config().sunny_batch_size]
        else:
            givens[obj.target_vars[key]] = ys_shared[key][idx*config().batch_size : (idx+1)*config().batch_size]

    for key in input_layers.keys():
        if key=="sunny":
            givens[input_layers[key].input_var] = xs_shared[key][idx*config().sunny_batch_size:(idx+1)*config().sunny_batch_size]
        else:
            givens[input_layers[key].input_var] = xs_shared[key][idx*config().batch_size:(idx+1)*config().batch_size]

    updates = config().build_updates(train_loss_theano, all_params, learning_rate)

    #grad_norm = T.sqrt(T.sum([(g**2).sum() for g in theano.grad(train_loss_theano, all_params)]))
    #theano_printer.print_me_this("Grad norm", grad_norm)

    iter_train = theano.function([idx], [train_loss_theano, kaggle_loss_theano, segmentation_loss_theano] + theano_printer.get_the_stuff_to_print(),
                                 givens=givens, on_unused_input="ignore", updates=updates,
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                 )
    iter_validate = theano.function([idx], [validation_train_loss, validation_kaggle_loss, validation_segmentation_loss] + [v for _, v in validation_other_losses.items()] + theano_printer.get_the_stuff_to_print(),
                                    givens=givens, on_unused_input="ignore")

    num_chunks_train = int(config().num_epochs_train * NUM_TRAIN_PATIENTS / (config().batch_size * config().batches_per_chunk))
    print "Will train for %d chunks" % num_chunks_train
    if config().restart_from_save and os.path.isfile(metadata_path):
        print "Load model parameters for resuming"
        resume_metadata = np.load(metadata_path)
        lasagne.layers.set_all_param_values(top_layer, resume_metadata['param_values'])
        start_chunk_idx = resume_metadata['chunks_since_start'] + 1
        chunks_train_idcs = range(start_chunk_idx, num_chunks_train)

        # set lr to the correct value
        current_lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
        print "  setting learning rate to %.7f" % current_lr
        learning_rate.set_value(current_lr)
        losses_train = resume_metadata['losses_train']
        losses_eval_valid = resume_metadata['losses_eval_valid']
        losses_eval_train = resume_metadata['losses_eval_train']
        losses_eval_valid_kaggle = [] #resume_metadata['losses_eval_valid_kaggle']
        losses_eval_train_kaggle = [] #resume_metadata['losses_eval_train_kaggle']
    else:
        chunks_train_idcs = range(num_chunks_train)
        losses_train = []
        losses_eval_valid = []
        losses_eval_train = []
        losses_eval_valid_kaggle = []
        losses_eval_train_kaggle = []


    create_train_gen = partial(config().create_train_gen,
                               required_input_keys = xs_shared.keys(),
                               required_output_keys = ys_shared.keys()# + ["patients"],
                               )


    create_eval_valid_gen = partial(config().create_eval_valid_gen,
                                   required_input_keys = xs_shared.keys(),
                                   required_output_keys = ys_shared.keys()# + ["patients"]
                                   )

    create_eval_train_gen = partial(config().create_eval_train_gen,
                                   required_input_keys = xs_shared.keys(),
                                   required_output_keys = ys_shared.keys()
                                   )

    print "Train model"
    start_time = time.time()
    prev_time = start_time

    num_batches_chunk = config().batches_per_chunk


    for e, train_data in izip(chunks_train_idcs, buffering.buffered_gen_threaded(create_train_gen())):
        print "Chunk %d/%d" % (e + 1, num_chunks_train)
        epoch = (1.0 * config().batch_size * config().batches_per_chunk * (e+1) / NUM_TRAIN_PATIENTS)
        print "  Epoch %.1f" % epoch

        for key, rate in learning_rate_schedule.iteritems():
            if epoch >= key:
                lr = np.float32(rate)
                learning_rate.set_value(lr)
        print "  learning rate %.7f" % lr

        if config().dump_network_loaded_data:
            pickle.dump(train_data, open("data_loader_dump_train_%d.pkl"%e, "wb"))

        for key in xs_shared:
            xs_shared[key].set_value(train_data["input"][key])

        for key in ys_shared:
            ys_shared[key].set_value(train_data["output"][key])

        #print "train:", sorted(train_data["output"]["patients"])
        losses = []
        kaggle_losses = []
        segmentation_losses = []
        for b in xrange(num_batches_chunk):
            iter_result = iter_train(b)

            loss, kaggle_loss, segmentation_loss = tuple(iter_result[:3])
            utils.detect_nans(loss, xs_shared, ys_shared, all_params)
 
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
            if config().validate_train_set:
                subsets = ["validation", "train"]
                gens = [create_eval_valid_gen, create_eval_train_gen]
                losses_eval = [losses_eval_valid, losses_eval_train]
                losses_kaggle = [losses_eval_valid_kaggle, losses_eval_train_kaggle]
            else:
                subsets = ["validation"]
                gens = [create_eval_valid_gen]
                losses_eval = [losses_eval_valid]
                losses_kaggle = [losses_eval_valid_kaggle]

            for subset, create_gen, losses_validation, losses_kgl in zip(subsets, gens, losses_eval, losses_kaggle):

                vld_losses = []
                vld_kaggle_losses = []
                vld_segmentation_losses = []
                vld_other_losses = {k:[] for k,_ in validation_other_losses.items()}
                print "  %s set (%d samples)" % (subset, get_number_of_validation_samples(set=subset))

                for validation_data in buffering.buffered_gen_threaded(create_gen()):
                    num_batches_chunk_eval = config().batches_per_chunk

                    if config().dump_network_loaded_data:
                        pickle.dump(validation_data, open("data_loader_dump_valid_%d.pkl"%e, "wb"))

                    for key in xs_shared:
                        xs_shared[key].set_value(validation_data["input"][key])

                    for key in ys_shared:
                        ys_shared[key].set_value(validation_data["output"][key])

                    #print "validate:", validation_data["output"]["patients"]

                    for b in xrange(num_batches_chunk_eval):
                        losses = tuple(iter_validate(b)[:3+len(validation_other_losses)])
                        loss, kaggle_loss, segmentation_loss = losses[:3]
                        other_losses = losses[3:]
                        vld_losses.extend(loss)
                        vld_kaggle_losses.extend(kaggle_loss)
                        vld_segmentation_losses.extend(segmentation_loss)
                        for k, other_loss in zip(validation_other_losses, other_losses):
                            vld_other_losses[k].extend(other_loss)

                vld_losses = np.array(vld_losses)
                vld_kaggle_losses = np.array(vld_kaggle_losses)
                vld_segmentation_losses = np.array(vld_segmentation_losses)
                for k in validation_other_losses:
                    vld_other_losses[k] = np.array(vld_other_losses[k])

                # now select only the relevant section to average
                sunny_len = get_lenght_of_set(name="sunny", set=subset)
                regular_len = get_lenght_of_set(name="regular", set=subset)
                num_valid_samples = get_number_of_validation_samples(set=subset)

                #print losses[:num_valid_samples]
                #print kaggle_losses[:regular_len]
                #print segmentation_losses[:sunny_len]
                loss_to_save = obj.compute_average(vld_losses[:num_valid_samples])
                print "  mean training loss:\t\t%.6f" % loss_to_save
                print "  mean kaggle loss:\t\t%.6f"   % np.mean(vld_kaggle_losses[:regular_len])
                print "  mean segment loss:\t\t%.6f"  % np.mean(vld_segmentation_losses[:sunny_len])
                # print "    acc:\t%.2f%%" % (acc * 100)
                for k, v in vld_other_losses.items():
                    print "  mean %s loss:\t\t%.6f"  % (k, obj.compute_average(v[:num_valid_samples], loss_name=k))
                print

                losses_validation.append(loss_to_save)

                kaggle_to_save = np.mean(vld_kaggle_losses[:regular_len])
                losses_kgl.append(kaggle_to_save)

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (float(num_chunks_train - (e + 1)) / float(e + 1 - chunks_train_idcs[0]))
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
                    'losses_eval_train': losses_eval_train,
                    'losses_eval_train_kaggle': losses_eval_train_kaggle,
                    'losses_eval_valid': losses_eval_valid,
                    'losses_eval_valid_kaggle': losses_eval_valid_kaggle,
                    'time_since_start': time_since_start,
                    'param_values': lasagne.layers.get_all_param_values(top_layer)
                }, f, pickle.HIGHEST_PROTOCOL)

            print "  saved to %s" % metadata_path
            print

    # store all known outputs from last batch:
    if config().take_a_dump:
        all_theano_variables = [train_loss_theano, kaggle_loss_theano, segmentation_loss_theano] + theano_printer.get_the_stuff_to_print()
        for layer in all_layers[:-1]:
            all_theano_variables.append(lasagne.layers.helper.get_output(layer))

        iter_train = theano.function([idx], all_theano_variables,
                                     givens=givens, on_unused_input="ignore", updates=updates,
                                     # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                     )
        train_data["intermediates"] = iter_train(0)
        pickle.dump(train_data, open(metadata_path + "-dump", "wb"))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)

    log_file = LOGS_PATH + "%s.log" % expid
    with print_to_file(log_file):

        print "Running configuration:", config().__name__
        print "Current git version:", utils.get_git_revision_hash()

        train_model(expid)
        print "log saved to '%s'" % log_file
        predict_model(expid)
        print "log saved to '%s'" % log_file


