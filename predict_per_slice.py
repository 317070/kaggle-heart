from __future__ import division

import argparse
import cPickle as pickle
import csv
import itertools
import string
import sys
import time

from datetime import timedelta, datetime
from functools import partial
from itertools import izip

import lasagne
import numpy as np
import theano
import theano.tensor as T

import buffering
import data_loader
import theano_printer
import utils

from configuration import config, set_configuration
from data_loader import get_number_of_test_batches, validation_patients_indices, train_patients_indices, regular_labels
from data_loader import NUM_PATIENTS
from postprocess import make_monotone_distribution, test_if_valid_distribution
from utils import CRSP


def _print_architecture(top_layer):
    all_layers = lasagne.layers.get_all_layers(top_layer)
    num_params = lasagne.layers.count_params(top_layer)
    print "  number of parameters: %d" % num_params
    print string.ljust("  layer output shapes:",36),
    print string.ljust("#params:",10),
    print "output shape:"
    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 32)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(num_param.__str__(), 10)
        print "    %s %s %s" % (name,  num_param, layer.output_shape)


def _check_slicemodel(input_layers):
    for inp in input_layers.keys():
        # If the input is image data but not a single slice
        if not inp.startswith("sliced:data:singleslice") and inp.startswith("sliced:data"):
            raise ValueError("predict_per_slice requires a slice model")


def predict_slice_model(expid, outfile, mfile=None):
    metadata_path = "/mnt/storage/metadata/kaggle-heart/train/%s.pkl" % (expid if not mfile else mfile)

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    print "Build model"
    interface_layers = config().build_model()

    output_layers = interface_layers["outputs"]
    input_layers = interface_layers["inputs"]
    top_layer = lasagne.layers.MergeLayer(
        incomings=output_layers.values()
    )
    _check_slicemodel(input_layers)

    # Print the architecture
    _print_architecture(top_layer)

    xs_shared = {
        key: lasagne.utils.shared_empty(dim=len(l_in.output_shape), dtype='float32') for (key, l_in) in input_layers.iteritems()
    }
    idx = T.lscalar('idx')

    givens = dict()

    for key in input_layers.keys():
        if key=="sunny":
            givens[input_layers[key].input_var] = xs_shared[key][idx*config().sunny_batch_size:(idx+1)*config().sunny_batch_size]
        else:
            givens[input_layers[key].input_var] = xs_shared[key][idx*config().batch_size:(idx+1)*config().batch_size]

    network_outputs = [
        lasagne.layers.helper.get_output(network_output_layer, deterministic=True)
        for network_output_layer in output_layers.values()
    ]

    iter_test = theano.function([idx], network_outputs + theano_printer.get_the_stuff_to_print(),
                                 givens=givens, on_unused_input="ignore",
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                 )

    print "Load model parameters for resuming"
    resume_metadata = np.load(metadata_path)
    lasagne.layers.set_all_param_values(top_layer, resume_metadata['param_values'])
    num_batches_chunk = config().batches_per_chunk
    num_batches = get_number_of_test_batches()
    num_chunks = int(np.ceil(num_batches / float(config().batches_per_chunk)))

    chunks_train_idcs = range(1, num_chunks+1)

    create_test_gen = partial(config().create_test_gen,
                              required_input_keys = xs_shared.keys(),
                              required_output_keys = ["patients", "slices"],
                              )

    print "Generate predictions with this model"
    start_time = time.time()
    prev_time = start_time


    predictions = [{"patient": i+1,
                    "slices": {
                        slice_id: {
                            "systole": np.zeros((0,600)),
                            "diastole": np.zeros((0,600))
                        } for slice_id in data_loader.get_slice_ids_for_patient(i+1)
                    }
                   } for i in xrange(NUM_PATIENTS)]


    # Loop over data and generate predictions
    for e, test_data in izip(itertools.count(start=1), buffering.buffered_gen_threaded(create_test_gen())):
        print "  load testing data onto GPU"

        for key in xs_shared:
            xs_shared[key].set_value(test_data["input"][key])


        patient_ids = test_data["output"]["patients"]
        slice_ids = test_data["output"]["slices"]
        print "  patients:", " ".join(map(str, patient_ids))
        print "  chunk %d/%d" % (e, num_chunks)

        for b in xrange(num_batches_chunk):
            iter_result = iter_test(b)
            network_outputs = tuple(iter_result[:len(output_layers)])
            network_outputs_dict = {output_layers.keys()[i]: network_outputs[i] for i in xrange(len(output_layers))}
            kaggle_systoles, kaggle_diastoles = config().postprocess(network_outputs_dict)
            kaggle_systoles, kaggle_diastoles = kaggle_systoles.astype('float64'), kaggle_diastoles.astype('float64')
            for idx, (patient_id, slice_id) in enumerate(
                    zip(patient_ids[b*config().batch_size:(b+1)*config().batch_size],
                        slice_ids[b*config().batch_size:(b+1)*config().batch_size])):
                if patient_id != 0:
                    index = patient_id-1
                    patient_data = predictions[index]
                    assert patient_id==patient_data["patient"]
                    patient_slice_data = patient_data["slices"][slice_id]
                    patient_slice_data["systole"] =  np.concatenate((patient_slice_data["systole"],  kaggle_systoles[idx:idx+1,:]),axis=0)
                    patient_slice_data["diastole"] = np.concatenate((patient_slice_data["diastole"], kaggle_diastoles[idx:idx+1,:]),axis=0)

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * (float(num_chunks - (e + 1)) / float(e + 1 - chunks_train_idcs[0]))
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
        print

    # Average predictions
    already_printed = False
    for prediction in predictions:
        for prediction_slice_id in prediction["slices"]:
            prediction_slice = prediction["slices"][prediction_slice_id]
            if prediction_slice["systole"].size>0 and prediction_slice["diastole"].size>0:
                average_method =  getattr(config(), 'tta_average_method', partial(np.mean, axis=0))
                prediction_slice["systole_average"] = average_method(prediction_slice["systole"])
                prediction_slice["diastole_average"] = average_method(prediction_slice["diastole"])
                try:
                    test_if_valid_distribution(prediction_slice["systole_average"])
                    test_if_valid_distribution(prediction_slice["diastole_average"])
                except:
                    if not already_printed:
                        print "WARNING: These distributions are not distributions"
                        already_printed = True
                    prediction_slice["systole_average"] = make_monotone_distribution(prediction_slice["systole_average"])
                    prediction_slice["diastole_average"] = make_monotone_distribution(prediction_slice["diastole_average"])


    print "Calculating training and validation set scores for reference"
    # Add CRPS scores to the predictions
    # Iterate over train and validation sets
    for patient_ids, set_name in [(validation_patients_indices, "validation"),
                                      (train_patients_indices,  "train")]:
        # Iterate over patients in the set
        for patient in patient_ids:
            prediction = predictions[patient-1]
            # Iterate over the slices
            for slice_id in prediction["slices"]:
                prediction_slice = prediction["slices"][slice_id]
                if "systole_average" in prediction_slice:
                    assert patient == regular_labels[patient-1, 0]
                    error_sys = CRSP(prediction_slice["systole_average"], regular_labels[patient-1, 1])
                    prediction_slice["systole_CRPS"] = error_sys
                    prediction_slice["target_systole"] = regular_labels[patient-1, 1]
                    error_dia = CRSP(prediction_slice["diastole_average"], regular_labels[patient-1, 2])
                    prediction_slice["diastole_CRPS"] = error_dia
                    prediction_slice["target_diastole"] = regular_labels[patient-1, 2]
                    prediction_slice["CRPS"] = 0.5 * error_sys + 0.5 * error_dia


    print "dumping prediction file to %s" % outfile
    with open(outfile, 'w') as f:
        pickle.dump({
                        'metadata_path': metadata_path,
                        'configuration_file': config().__name__,
                        'git_revision_hash': utils.get_git_revision_hash(),
                        'experiment_id': expid,
                        'time_since_start': time_since_start,
                        'param_values': lasagne.layers.get_all_param_values(top_layer),
                        'predictions_per_slice': predictions,
                    }, f, pickle.HIGHEST_PROTOCOL)
    print "prediction file dumped"


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    required.add_argument('-o', '--output',
                          help='output file',
                          required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-m', '--metadata',
                          help='metadatafile to use',
                          required=False)

    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)
    mfile = args.metadata
    ofile = args.output

    predict_slice_model(expid, ofile, mfile)
