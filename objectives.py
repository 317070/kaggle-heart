import numpy as np
import lasagne
import theano
import theano.tensor as T
import theano_printer

class TargetVarDictObjective(object):
    def __init__(self, input_layers, penalty=0):
        try:
            self.target_vars
        except:
            self.target_vars = dict()
        self.penalty = penalty

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_kaggle_loss(self, *args, **kwargs):
        return theano.shared([-1])

    def get_segmentation_loss(self, *args, **kwargs):
        return theano.shared([-1])


class KaggleObjective(TargetVarDictObjective):
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/second-annual-data-science-bowl/details/evaluation
    """
    def __init__(self, input_layers, *args, **kwargs):
        super(KaggleObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole = input_layers["systole"]
        self.input_diastole = input_layers["diastole"]

        self.target_vars["systole"]  = T.fmatrix("systole_target")
        self.target_vars["diastole"] = T.fmatrix("diastole_target")

    def get_loss(self, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]

        if "average" in kwargs and not kwargs["average"]:
            CRPS = 0.5 * T.mean((network_systole - systole_target)**2,  axis = (1,)) + \
                   0.5 * T.mean((network_diastole - diastole_target)**2, axis = (1,))
            return CRPS

        CRPS = 0.5 * T.mean((network_systole - systole_target)**2,  axis = (0,1)) + \
               0.5 * T.mean((network_diastole - diastole_target)**2, axis = (0,1))
        return CRPS + self.penalty


class MSObjective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(RMSObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole = input_layers["systole:value"]
        self.input_diastole = input_layers["diastole:value"]

        self.target_vars["systole:value"]  = T.fvector("systole_target_value")
        self.target_vars["diastole:value"] = T.fvector("diastole_target_value")

    def get_loss(self, average=True, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)[:,0]
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)[:,0]

        systole_target = self.target_vars["systole:value"]
        diastole_target = self.target_vars["diastole:value"]

        if not average:
            # The following is not strictly correct
            loss = 0.5 * (network_systole  - systole_target )**2 + \
                   0.5 * (network_diastole - diastole_target)**2
            return loss

        loss = T.mean(0.5 * (network_systole  - systole_target )**2 + \
                      0.5 * (network_diastole - diastole_target)**2, axis=(0,)
                             )
        return loss + self.penalty


class KaggleValidationMSObjective(MSObjective):
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/second-annual-data-science-bowl/details/evaluation
    """
    def __init__(self, input_layers, *args, **kwargs):
        super(KaggleValidationRMSObjective, self).__init__(input_layers, *args, **kwargs)
        self.target_vars["systole"]  = T.fmatrix("systole_target_kaggle")
        self.target_vars["diastole"] = T.fmatrix("diastole_target_kaggle")

    def get_kaggle_loss(self, validation=False, average=True, *args, **kwargs):
        if not validation:  # only evaluate this one in the validation step
            return theano.shared([-1])

        sigma = T.sqrt(self.get_loss() - self.penalty)

        def theano_mu_sigma_erf(mu_erf, sigma_erf):
            eps = 1e-7
            x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x',0)
            sigma_erf = T.clip(sigma_erf.dimshuffle('x','x'), eps, 1)
            x = (x_axis - mu_erf.dimshuffle(0,'x')) / (sigma_erf * np.sqrt(2).astype('float32'))
            return (T.erf(x) + 1)/2

        network_systole  = theano_mu_sigma_erf(lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)[:,0],
                                                                                sigma)
        network_diastole = theano_mu_sigma_erf(lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)[:,0],
                                                                                sigma)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]

        if not average:
            CRPS = (T.mean((network_systole - systole_target)**2,  axis = (1,)) +
                    T.mean((network_diastole - diastole_target)**2, axis = (1,)) )/2
            return CRPS
        else:
            CRPS = (T.mean((network_systole - systole_target)**2,  axis = (0,1)) +
                    T.mean((network_diastole - diastole_target)**2, axis = (0,1)) )/2
            return CRPS



class LogLossObjective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(LogLossObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole = input_layers["systole:onehot"]
        self.input_diastole = input_layers["diastole:onehot"]

        self.target_vars["systole:onehot"]  = T.fmatrix("systole_target_onehot")
        self.target_vars["diastole:onehot"] = T.fmatrix("diastole_target_onehot")

    def get_loss(self, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole:onehot"]
        diastole_target = self.target_vars["diastole:onehot"]

        if "average" in kwargs and not kwargs["average"]:
            ll = 0.5 * log_loss(network_systole, systole_target) + \
                 0.5 * log_loss(network_diastole, diastole_target)
            return ll

        ll = 0.5 * T.mean(log_loss(network_systole, systole_target),  axis = (0,)) + \
             0.5 * T.mean(log_loss(network_diastole, diastole_target), axis = (0,))
        return ll + self.penalty


class KaggleValidationLogLossObjective(LogLossObjective):
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/second-annual-data-science-bowl/details/evaluation
    """
    def __init__(self, input_layers, *args, **kwargs):
        super(KaggleValidationLogLossObjective, self).__init__(input_layers, *args, **kwargs)
        self.target_vars["systole"]  = T.fmatrix("systole_target_kaggle")
        self.target_vars["diastole"] = T.fmatrix("diastole_target_kaggle")

    def get_kaggle_loss(self, validation=False, average=True, *args, **kwargs):
        if not validation:
            return theano.shared([-1])

        network_systole  = T.clip(T.extra_ops.cumsum(lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs), axis=1), 0.0, 1.0)
        network_diastole = T.clip(T.extra_ops.cumsum(lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs), axis=1), 0.0, 1.0)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]

        if not average:
            CRPS = (T.mean((network_systole - systole_target)**2,  axis = (1,)) +
                    T.mean((network_diastole - diastole_target)**2, axis = (1,)) )/2
            return CRPS
        else:
            CRPS = (T.mean((network_systole - systole_target)**2,  axis = (0,1)) +
                    T.mean((network_diastole - diastole_target)**2, axis = (0,1)) )/2
            return CRPS


class BinaryCrossentropyImageObjective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(BinaryCrossentropyImageObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_layer = input_layers["segmentation"]
        self.target_vars = dict()
        self.target_vars["segmentation"] = T.ftensor3("segmentation_target")

    def get_loss(self, *args, **kwargs):
        network_output = lasagne.layers.helper.get_output(self.input_layer, *args, **kwargs)
        segmentation_target = self.target_vars["segmentation"]

        if "average" in kwargs and not kwargs["average"]:
            loss = log_loss( network_output.flatten(ndim=2), segmentation_target.flatten(ndim=2) )
            return loss

        return T.mean(log_loss(network_output.flatten(ndim=2), segmentation_target.flatten(ndim=2))) + self.penalty


class MixedKaggleSegmentationObjective(KaggleObjective, BinaryCrossentropyImageObjective):
    def __init__(self, input_layers, segmentation_weight=1.0, *args, **kwargs):
        super(MixedKaggleSegmentationObjective, self).__init__(input_layers, *args, **kwargs)
        self.segmentation_weight = segmentation_weight

    def get_loss(self, *args, **kwargs):
        return self.get_kaggle_loss(*args, **kwargs) + self.segmentation_weight * self.get_segmentation_loss(*args, **kwargs)

    def get_kaggle_loss(self, *args, **kwargs):
        return KaggleObjective.get_loss(self, *args, **kwargs)

    def get_segmentation_loss(self, *args, **kwargs):
        return BinaryCrossentropyImageObjective.get_loss(self, *args, **kwargs)



class UpscaledImageObjective(BinaryCrossentropyImageObjective):
    def get_loss(self, *args, **kwargs):
        network_output = lasagne.layers.helper.get_output(self.input_layer, *args, **kwargs)
        segmentation_target = self.target_vars["segmentation"]
        return log_loss(network_output.flatten(ndim=2), segmentation_target[:,4::8,4::8].flatten(ndim=2)) + self.penalty

def log_loss(y, t, eps=1e-7):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.mean(t * np.log(y) + (1-t) * np.log(1-y), axis=(1,))
    return loss


class R2Objective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(R2Objective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole = input_layers["systole"]
        self.input_diastole = input_layers["diastole"]
        self.target_vars["systole"] = T.fvector("systole_target")
        self.target_vars["diastole"] = T.fvector("diastole_target")

    def get_loss(self, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]
        return T.sum((network_diastole-diastole_target)**2) + T.sum((network_systole-systole_target)**2) + self.penalty

