"""Script for generating predictions for a given trained model.

The script loads the specified configuration file. All parameters are defined
in that file.

Usage:
> python predict.py -c CONFIG_NAME
"""

from __future__ import division

import argparse
import cPickle as pickle
import csv
import itertools
import string
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
from paths import MODEL_PATH
from paths import INTERMEDIATE_PREDICTIONS_PATH
from paths import SUBMISSION_PATH
from postprocess import make_monotone_distribution, test_if_valid_distribution
from utils import CRSP

def predict_model(expid, mfile=None):
    metadata_path = MODEL_PATH + "%s.pkl" % (expid if not mfile else mfile)
    prediction_path = INTERMEDIATE_PREDICTIONS_PATH + "%s.pkl" % expid
    submission_path = SUBMISSION_PATH + "%s.csv" % expid

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    print "Using"
    print "  %s" % metadata_path
    print "To generate"
    print "  %s" % prediction_path
    print "  %s" % submission_path

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
    print string.ljust("  layer output shapes:",36),
    print string.ljust("#params:",10),
    print "output shape:"
    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 32)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(num_param.__str__(), 10)
        print "    %s %s %s" % (name,  num_param, layer.output_shape)

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

    data_loader.filter_patient_folders()

    create_test_gen = partial(config().create_test_gen,
                              required_input_keys = xs_shared.keys(),
                              required_output_keys = ["patients", "classification_correction_function"],
                              )

    print "Generate predictions with this model"
    start_time = time.time()
    prev_time = start_time


    predictions = [{"patient": i+1,
                    "systole": np.zeros((0,600)),
                    "diastole": np.zeros((0,600))
                    } for i in xrange(NUM_PATIENTS)]


    for e, test_data in izip(itertools.count(start=1), buffering.buffered_gen_threaded(create_test_gen())):
        print "  load testing data onto GPU"

        for key in xs_shared:
            xs_shared[key].set_value(test_data["input"][key])


        patient_ids = test_data["output"]["patients"]
        classification_correction = test_data["output"]["classification_correction_function"]
        print "  patients:", " ".join(map(str, patient_ids))
        print "  chunk %d/%d" % (e, num_chunks)

        for b in xrange(num_batches_chunk):
            iter_result = iter_test(b)
            network_outputs = tuple(iter_result[:len(output_layers)])
            network_outputs_dict = {output_layers.keys()[i]: network_outputs[i] for i in xrange(len(output_layers))}
            kaggle_systoles, kaggle_diastoles = config().postprocess(network_outputs_dict)
            kaggle_systoles, kaggle_diastoles = kaggle_systoles.astype('float64'), kaggle_diastoles.astype('float64')
            for idx, patient_id in enumerate(patient_ids[b*config().batch_size:(b+1)*config().batch_size]):
                if patient_id != 0:
                    index = patient_id-1
                    patient_data = predictions[index]
                    assert patient_id==patient_data["patient"]

                    kaggle_systole = kaggle_systoles[idx:idx+1,:]
                    kaggle_diastole = kaggle_diastoles[idx:idx+1,:]
                    assert np.isfinite(kaggle_systole).all() and np.isfinite(kaggle_systole).all()
                    kaggle_systole = classification_correction[b*config().batch_size + idx](kaggle_systole)
                    kaggle_diastole = classification_correction[b*config().batch_size + idx](kaggle_diastole)
                    assert np.isfinite(kaggle_systole).all() and np.isfinite(kaggle_systole).all()
                    patient_data["systole"] =  np.concatenate((patient_data["systole"], kaggle_systole ),axis=0)
                    patient_data["diastole"] = np.concatenate((patient_data["diastole"], kaggle_diastole ),axis=0)

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

    already_printed = False
    for prediction in predictions:
        if prediction["systole"].size>0 and prediction["diastole"].size>0:
            average_method =  getattr(config(), 'tta_average_method', partial(np.mean, axis=0))
            prediction["systole_average"] = average_method(prediction["systole"])
            prediction["diastole_average"] = average_method(prediction["diastole"])
            try:
                test_if_valid_distribution(prediction["systole_average"])
                test_if_valid_distribution(prediction["diastole_average"])
            except:
                if not already_printed:
                    print "WARNING: These distributions are not distributions"
                    already_printed = True
                prediction["systole_average"] = make_monotone_distribution(prediction["systole_average"])
                prediction["diastole_average"] = make_monotone_distribution(prediction["diastole_average"])
                test_if_valid_distribution(prediction["systole_average"])
                test_if_valid_distribution(prediction["diastole_average"])


    print "Calculating training and validation set scores for reference"

    validation_dict = {}
    for patient_ids, set_name in [(validation_patients_indices, "validation"),
                                      (train_patients_indices,  "train")]:
        errors = []
        for patient in patient_ids:
            prediction = predictions[patient-1]
            if "systole_average" in prediction:
                assert patient == regular_labels[patient-1, 0]
                error = CRSP(prediction["systole_average"], regular_labels[patient-1, 1])
                errors.append(error)
                error = CRSP(prediction["diastole_average"], regular_labels[patient-1, 2])
                errors.append(error)
        if len(errors)>0:
            errors = np.array(errors)
            estimated_CRSP = np.mean(errors)
            print "  %s kaggle loss: %f" % (string.rjust(set_name, 12), estimated_CRSP)
            validation_dict[set_name] = estimated_CRSP
        else:
            print "  %s kaggle loss: not calculated" % (string.rjust(set_name, 12))


    print "dumping prediction file to %s" % prediction_path
    with open(prediction_path, 'w') as f:
        pickle.dump({
                        'metadata_path': metadata_path,
                        'prediction_path': prediction_path,
                        'submission_path': submission_path,
                        'configuration_file': config().__name__,
                        'git_revision_hash': utils.get_git_revision_hash(),
                        'experiment_id': expid,
                        'time_since_start': time_since_start,
                        'param_values': lasagne.layers.get_all_param_values(top_layer),
                        'predictions': predictions,
                        'validation_errors': validation_dict,
                    }, f, pickle.HIGHEST_PROTOCOL)
    print "prediction file dumped"

    print "dumping submission file to %s" % submission_path
    with open(submission_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Id'] + ['P%d'%i for i in xrange(600)])
        for prediction in predictions:
            # the submission only has patients 501 to 700
            if prediction["patient"] in data_loader.test_patients_indices:
                if "diastole_average" not in prediction or "systole_average" not in prediction:
                    raise Exception("Not all test-set patients were predicted")
                csvwriter.writerow(["%d_Diastole" % prediction["patient"]] + ["%.18f" % p for p in prediction["diastole_average"].flatten()])
                csvwriter.writerow(["%d_Systole" % prediction["patient"]] + ["%.18f" % p for p in prediction["systole_average"].flatten()])
    print "submission file dumped"

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-m', '--metadata',
                          help='metadatafile to use',
                          required=False)

    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)
    mfile = args.metadata

    predict_model(expid, mfile)
