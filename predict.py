from __future__ import division
import argparse
from functools import partial
from itertools import izip
from datetime import timedelta, datetime
import lasagne
import theano
import time
import buffering
from configuration import config, set_configuration
import numpy as np
import string
from data_loader import get_number_of_test_batches
import theano_printer
import utils
import theano.tensor as T
import cPickle as pickle
import csv

def predict_model(expid):
    metadata_path = "/mnt/storage/metadata/kaggle-heart/train/%s.pkl" % expid
    prediction_path = "/mnt/storage/metadata/kaggle-heart/predictions/%s.pkl" % expid
    submission_path = "/mnt/storage/metadata/kaggle-heart/submissions/%s.csv" % expid

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
    num_batches = get_number_of_test_batches(set="test")
    num_chunks = int(np.ceil(num_batches / float(config().batches_per_chunk)))

    chunks_train_idcs = range(1, num_chunks+1)

    create_test_gen = partial(config().create_test_gen,
                              required_input_keys = xs_shared.keys(),
                              required_output_keys = ["patients"],
                              )

    print "Generate predictions with this model"
    start_time = time.time()
    prev_time = start_time


    predictions = [{"patient": i+501,
                    "systole":np.zeros((0,600)),
                    "diastole":np.zeros((0,600))
                    } for i in xrange(200)]


    #for e, test_data in izip(chunks_train_idcs, buffering.buffered_gen_threaded(create_test_gen())):
    for e, test_data in izip(chunks_train_idcs, create_test_gen()):
        print "  load training data onto GPU"

        for key in xs_shared:
            xs_shared[key].set_value(test_data["input"][key])


        patient_ids = test_data["output"]["patients"]
        print patient_ids
        print "  chunk %d/%d" % (e, num_chunks)

        for b in xrange(num_batches_chunk):
            iter_result = iter_test(b)
            network_outputs = tuple(iter_result[:len(output_layers)])
            network_outputs_dict = {output_layers.keys()[i]: network_outputs[i] for i in xrange(len(output_layers))}
            kaggle_systoles, kaggle_diastoles = config().postprocess(network_outputs_dict)

            for idx, patient_id in enumerate(patient_ids[b*config().batch_size:(b+1)*config().batch_size]):
                if 500 < patient_id <= 700:
                    index = patient_id-501
                    patient_data = predictions[index]
                    assert patient_id==patient_data["patient"]
                    patient_data["systole"] = np.concatenate((patient_data["systole"], kaggle_systoles[idx:idx+1,:]),axis=0)
                    patient_data["diastole"] = np.concatenate((patient_data["diastole"], kaggle_systoles[idx:idx+1,:]),axis=0)

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

    for prediction in predictions:
        prediction["systole_average"] = np.mean(prediction["systole"], axis=0)
        prediction["diastole_average"] = np.mean(prediction["diastole"], axis=0)

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
                    }, f, pickle.HIGHEST_PROTOCOL)
    print "prediction file dumped"

    print "dumping submission file to %s" % submission_path
    with open(submission_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Id'] + ['P%d'%i for i in xrange(600)])
        for prediction in predictions:
            # the submission only has patients 501 to 700
            if 500 < prediction["patient"] <= 700:
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

    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)

    predict_model(expid)
