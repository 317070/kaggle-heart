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
import theano_printer
import utils
import theano.tensor as T

def predict_model(metadata_path, metadata=None):

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"
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

    iter_test = theano.function([idx], [output_layers.values()] + theano_printer.get_the_stuff_to_print(),
                                 givens=givens, on_unused_input="ignore",
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                 )

    print "Load model parameters for resuming"
    resume_metadata = np.load(metadata_path)
    lasagne.layers.set_all_param_values(top_layer, resume_metadata['param_values'])
    start_chunk_idx = resume_metadata['chunks_since_start'] + 1
    chunks_train_idcs = range(start_chunk_idx, config().num_chunks_train)

    create_test_gen = partial(config().create_test_gen,
                              required_input_keys = xs_shared.keys(),
                              )

    print "Train model"
    start_time = time.time()
    prev_time = start_time

    num_batches_chunk = config().batches_per_chunk

    for e, test_data in izip(chunks_train_idcs, buffering.buffered_gen_threaded(create_test_gen())):
        print "Chunk %d/%d" % (e + 1, config().num_chunks_train)
        print "  load training data onto GPU"

        for key in xs_shared:
            xs_shared[key].set_value(test_data["input"][key])

        print "  batch SGD"
        losses = []
        kaggle_losses = []
        segmentation_losses = []
        for b in xrange(num_batches_chunk):
            sample_id = test_data["id"]
            iter_result = iter_test(b)

            loss, kaggle_loss, segmentation_loss = tuple(iter_result[:3])

            losses.append(loss)
            kaggle_losses.append(kaggle_loss)
            segmentation_losses.append(segmentation_loss)

        mean_train_loss = np.mean(losses)
        print "  mean training loss:\t\t%.6f" % mean_train_loss
        losses_train.append(mean_train_loss)

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

    # store all known outputs from last batch:

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)

    required.add_argument('-m', '--model',
                          help='model file with metadata',
                          required=True)

    args = parser.parse_args()
    set_configuration(args.config)

    predict_model(args.model)
