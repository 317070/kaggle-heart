import numpy as np
import lasagne
import theano
import theano.tensor as T

def binary_crossentropy_image_objective(predictions, targets):
    return log_loss(predictions, targets)

class BinaryCrossentropyImageObjective(object):
    def __init__(self, input_layers):
        self.input_layer = input_layers["segmentation"]
        self.target_vars = dict()
        self.target_vars["segmentation"] = T.ftensor3("segmentation_target")

    def get_loss(self, *args, **kwargs):
        network_output = lasagne.layers.helper.get_output(self.input_layer, *args, **kwargs)
        return log_loss(network_output, self.target_vars["segmentation"])

class UpscaledImageObjective(BinaryCrossentropyImageObjective):
    def get_loss(self, *args, **kwargs):
        network_output = lasagne.layers.helper.get_output(self.input_layer, *args, **kwargs)
        segmentation_target = self.target_vars["segmentation"]
        return log_loss(network_output, segmentation_target[:,4::8,4::8].flatten(ndim=2))

def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.mean(t * np.log(y) + (1-t) * np.log(1-y))
    return loss



class R2Objective(BinaryCrossentropyImageObjective):
    def __init__(self, input_layers):
        self.input_systole = input_layers["systole"]
        self.input_diastole = input_layers["diastole"]
        self.target_vars = dict()
        self.target_vars["systole"] = T.fvector("systole_target")
        self.target_vars["diastole"] = T.fvector("diastole_target")

    def get_loss(self, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]
        return T.sum((network_diastole-diastole_target)**2) + T.sum((network_systole-systole_target)**2)

