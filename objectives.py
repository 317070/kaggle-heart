import numpy as np
import lasagne
import theano
import theano.tensor as T
import theano_printer

class TargetVarDictObjective(object):
    def __init__(self, input_layers):
        try:
            self.target_vars
        except:
            self.target_vars = dict()

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_kaggle_loss(self, *args, **kwargs):
        return -1

    def get_segmentation_loss(self, *args, **kwargs):
        return -1


class BinaryCrossentropyImageObjective(TargetVarDictObjective):
    def __init__(self, input_layers):
        super(BinaryCrossentropyImageObjective, self).__init__(input_layers)
        self.input_layer = input_layers["segmentation"]
        self.target_vars = dict()
        self.target_vars["segmentation"] = T.ftensor3("segmentation_target")

    def get_loss(self, *args, **kwargs):
        network_output = lasagne.layers.helper.get_output(self.input_layer, *args, **kwargs)
        segmentation_target = self.target_vars["segmentation"]

        if "average" in kwargs and not kwargs["average"]:
            loss = log_loss( network_output.flatten(ndim=2), segmentation_target.flatten(ndim=2) )
            return loss

        return T.mean(log_loss(network_output.flatten(ndim=2), segmentation_target.flatten(ndim=2)))


class UpscaledImageObjective(BinaryCrossentropyImageObjective):
    def get_loss(self, *args, **kwargs):
        network_output = lasagne.layers.helper.get_output(self.input_layer, *args, **kwargs)
        segmentation_target = self.target_vars["segmentation"]
        return log_loss(network_output.flatten(ndim=2), segmentation_target[:,4::8,4::8].flatten(ndim=2))

def log_loss(y, t, eps=1e-7):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.mean(t * np.log(y) + (1-t) * np.log(1-y), axis=(1,))
    return loss


class R2Objective(TargetVarDictObjective):
    def __init__(self, input_layers):
        super(R2Objective, self).__init__(input_layers)
        self.input_systole = input_layers["systole"]
        self.input_diastole = input_layers["diastole"]
        self.target_vars["systole"] = T.fvector("systole_target")
        self.target_vars["diastole"] = T.fvector("diastole_target")

    def get_loss(self, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]
        return T.sum((network_diastole-diastole_target)**2) + T.sum((network_systole-systole_target)**2)


class KaggleObjective(TargetVarDictObjective):
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/second-annual-data-science-bowl/details/evaluation
    """
    def __init__(self, input_layers):
        super(KaggleObjective, self).__init__(input_layers)
        self.input_systole = input_layers["systole"]
        self.input_diastole = input_layers["diastole"]

        self.target_vars["systole"]  = T.fmatrix("systole_target")
        self.target_vars["diastole"] = T.fmatrix("diastole_target")

    def get_loss(self, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]

        P_systole = T.extra_ops.cumsum(network_systole, axis=1)
        P_diastole = T.extra_ops.cumsum(network_diastole, axis=1)

        if "average" in kwargs and not kwargs["average"]:
            CRPS = 0.5 * T.mean((P_systole - systole_target)**2,  axis = (1,)) + \
                   0.5 * T.mean((P_diastole - diastole_target)**2, axis = (1,))
            return CRPS

        CRPS = 0.5 * T.mean((P_systole - systole_target)**2,  axis = (0,1)) + \
               0.5 * T.mean((P_diastole - diastole_target)**2, axis = (0,1))
        return CRPS


class MixedKaggleSegmentationObjective(KaggleObjective, BinaryCrossentropyImageObjective):
    def __init__(self, input_layers):
        super(MixedKaggleSegmentationObjective, self).__init__(input_layers)

    def get_loss(self, *args, **kwargs):
        return self.get_kaggle_loss(*args, **kwargs) + 0*self.get_segmentation_loss(*args, **kwargs)

    def get_kaggle_loss(self, *args, **kwargs):
        return KaggleObjective.get_loss(self, *args, **kwargs)

    def get_segmentation_loss(self, *args, **kwargs):
        return BinaryCrossentropyImageObjective.get_loss(self, *args, **kwargs)
