import sys
import numpy as np
import theano
import theano.tensor as T
from itertools import izip
import lasagne as nn
import time
import os
import string
import importlib
import utils

if not (3 <= len(sys.argv) <= 5):
    sys.exit("Usage: test.py <configuration_name> <metadata_path>")

config_name = sys.argv[1]
metadata_path = sys.argv[2]

print "Load parameters"
metadata = np.load(metadata_path)

if config_name == "_":
    config_name = metadata['configuration']

config = importlib.import_module("configurations.%s" % config_name)

filename = os.path.splitext(os.path.basename(metadata_path))[0]
target_path = "predictions/%s--%s.npy" % (config_name, filename)

print "Build model"
model = config.build_model()
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

nn.layers.set_all_param_values(model.l_top, metadata['param_values'])

test_data_iterator = config().test_data_iterator

print
print 'Data'
print 'n test: %d' % test_data_iterator.nsamples

xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
givens_in = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_in[l_in.input_var] = x

iter_test = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_outs], givens=givens_in)



for iter_idx, (xs_batch, ys_batch, _) in izip(iter_idxs, buffering.buffered_gen_threaded(train_data_iterator.generate())):

print "Saving"
np.save(target_path, outputs)
print "  saved to %s" % target_path
