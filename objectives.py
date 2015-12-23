import numpy as np
import lasagne
import theano
import theano.tensor as T

def binary_crossentropy_image_objective(predictions, targets):
    return log_loss(predictions, targets)

class BinaryCrossentropyImageObjective(object):
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.target_var = T.matrix("target")

    def get_loss(self, input=None, target=None, *args, **kwargs):
        network_output = lasagne.layers.helper.get_output(self.input_layer, *args, **kwargs)
        return log_loss(network_output, target)

def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
    return loss