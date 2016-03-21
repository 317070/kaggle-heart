"""Library implementing different objective functions.
"""

import numpy as np
import lasagne
import theano
import theano.tensor as T

import theano_printer
import utils


class TargetVarDictObjective(object):
    def __init__(self, input_layers, penalty=0):
        try:
            self.target_vars
        except:
            self.target_vars = dict()
        self.penalty = penalty

    def get_loss(self, average=True, *args, **kwargs):
        """Compute the loss in Theano.

        Args:
            average: Indicates whether the loss should already be averaged over the batch.
                If not, call the compute_average method on the aggregated losses.
        """
        raise NotImplementedError

    def compute_average(self, losses, loss_name=""):
        """Averages the aggregated losses in Numpy."""
        return losses.mean(axis=0)

    def get_kaggle_loss(self, average=True, *args, **kwargs):
        """Computes the CRPS score in Theano."""
        return theano.shared([-1])

    def get_segmentation_loss(self, average=True, *args, **kwargs):
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

    def get_loss(self, average=True, other_losses={}, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]

        CRPS_systole = T.mean((network_systole - systole_target)**2, axis=(1,))
        CRPS_diastole = T.mean((network_diastole - diastole_target)**2, axis=(1,))
        loss = 0.5*CRPS_systole + 0.5*CRPS_diastole

        if average:
            loss = T.mean(loss, axis=(0,))
            CRPS_systole = T.mean(CRPS_systole, axis=(0,))
            CRPS_diastole = T.mean(CRPS_diastole, axis=(0,))

        other_losses['CRPS_systole'] = CRPS_systole
        other_losses['CRPS_diastole'] = CRPS_diastole
        return loss + self.penalty

    #def get_kaggle_loss(self, *args, **kwargs):
    #    return self.get_loss(*args, **kwargs)


class MeanKaggleObjective(TargetVarDictObjective):
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/second-annual-data-science-bowl/details/evaluation
    """
    def __init__(self, input_layers, *args, **kwargs):
        super(MeanKaggleObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_average = input_layers["average"]
        self.target_vars["average"]  = T.fmatrix("average_target")
        self.input_systole = input_layers["systole"]
        self.input_diastole = input_layers["diastole"]

        self.target_vars["systole"]  = T.fmatrix("systole_target")
        self.target_vars["diastole"] = T.fmatrix("diastole_target")

    def get_loss(self, average=True, other_losses={}, *args, **kwargs):
        network_average  = lasagne.layers.helper.get_output(self.input_average, *args, **kwargs)
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        average_target = self.target_vars["average"]
        systole_target = self.target_vars["systole"]
        diastole_target = self.target_vars["diastole"]

        CRPS_average = T.mean((network_average - average_target)**2, axis=(1,))
        CRPS_systole = T.mean((network_systole - systole_target)**2, axis=(1,))
        CRPS_diastole = T.mean((network_diastole - diastole_target)**2, axis=(1,))
        loss = 0.2*CRPS_average + 0.4*CRPS_systole + 0.4*CRPS_diastole

        if average:
            loss = T.mean(loss, axis=(0,))
            CRPS_average = T.mean(CRPS_average, axis=(0,))
            CRPS_systole = T.mean(CRPS_systole, axis=(0,))
            CRPS_diastole = T.mean(CRPS_diastole, axis=(0,))

        other_losses['CRPS_average'] = CRPS_average
        other_losses['CRPS_systole'] = CRPS_systole
        other_losses['CRPS_diastole'] = CRPS_diastole
        return loss + self.penalty

    #def get_kaggle_loss(self, *args, **kwargs):
    #    return self.get_loss(*args, **kwargs)


class MSEObjective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(MSEObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole = input_layers["systole:value"]
        self.input_diastole = input_layers["diastole:value"]

        self.target_vars["systole:value"]  = T.fvector("systole_target_value")
        self.target_vars["diastole:value"] = T.fvector("diastole_target_value")

    def get_loss(self, average=True, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)[:,0]
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)[:,0]

        systole_target = self.target_vars["systole:value"]
        diastole_target = self.target_vars["diastole:value"]

        loss = 0.5 * (network_systole  - systole_target )**2 + 0.5 * (network_diastole - diastole_target)**2

        if average:
            loss = T.mean(loss, axis=(0,))
        return loss + self.penalty


class RMSEObjective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(RMSEObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole = input_layers["systole:value"]
        self.input_diastole = input_layers["diastole:value"]

        self.target_vars["systole:value"] = T.fvector("systole_target_value")
        self.target_vars["diastole:value"] = T.fvector("diastole_target_value")

    def get_loss(self, average=True, *args, **kwargs):
        network_systole = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)[:,0]
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)[:,0]

        systole_target = self.target_vars["systole:value"]
        diastole_target = self.target_vars["diastole:value"]

        loss = 0.5 * (network_systole - systole_target) ** 2 + 0.5 * (network_diastole - diastole_target)**2

        if average:
            loss = T.sqrt(T.mean(loss, axis=(0,)))
        return loss

    def compute_average(self, aggregate):
        return np.sqrt(np.mean(aggregate, axis=0))

    def get_kaggle_loss(self, validation=False, average=True, *args, **kwargs):
        if not validation:  # only evaluate this one in the validation step
            return theano.shared([-1])

        network_systole  = utils.theano_mu_sigma_erf(lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)[:,0],
                                                     lasagne.layers.helper.get_output(self.input_systole_sigma, *args, **kwargs)[:,0])
        network_diastole = utils.theano_mu_sigma_erf(lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)[:,0],
                                                     lasagne.layers.helper.get_output(self.input_diastole_sigma, *args, **kwargs)[:,0])

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


class KaggleValidationMSEObjective(MSEObjective):
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/second-annual-data-science-bowl/details/evaluation
    """
    def __init__(self, input_layers, *args, **kwargs):
        super(KaggleValidationMSEObjective, self).__init__(input_layers, *args, **kwargs)
        self.target_vars["systole"]  = T.fmatrix("systole_target_kaggle")
        self.target_vars["diastole"] = T.fmatrix("diastole_target_kaggle")

    def get_kaggle_loss(self, validation=False, average=True, *args, **kwargs):
        if not validation:  # only evaluate this one in the validation step
            return theano.shared([-1])

        sigma = T.sqrt(self.get_loss() - self.penalty)

        network_systole  = utils.theano_mu_sigma_erf(lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)[:,0],
                                                                                sigma)
        network_diastole = utils.theano_mu_sigma_erf(lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)[:,0],
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


def _theano_pdf_to_cdf(pdfs):
    return T.extra_ops.cumsum(pdfs, axis=1)


def _crps(cdfs1, cdfs2):
    return T.mean((cdfs1 - cdfs2)**2, axis=(1,))


class LogLossObjective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(LogLossObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole = input_layers["systole:onehot"]
        self.input_diastole = input_layers["diastole:onehot"]

        self.target_vars["systole:onehot"]  = T.fmatrix("systole_target_onehot")
        self.target_vars["diastole:onehot"] = T.fmatrix("diastole_target_onehot")

    def get_loss(self, average=True, other_losses={}, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole:onehot"]
        diastole_target = self.target_vars["diastole:onehot"]

        ll_sys = log_loss(network_systole, systole_target)
        ll_dia = log_loss(network_diastole, diastole_target)
        ll = 0.5 * ll_sys + 0.5 * ll_dia

        # CRPS scores
        cdf = _theano_pdf_to_cdf
        CRPS_systole = _crps(cdf(network_systole), cdf(systole_target))
        CRPS_diastole = _crps(cdf(network_diastole), cdf(diastole_target))

        if average:
            ll = T.mean(ll, axis=(0,))
            CRPS_systole = T.mean(CRPS_systole, axis=(0,))
            CRPS_diastole = T.mean(CRPS_diastole, axis=(0,))

        other_losses['CRPS_systole'] = CRPS_systole
        other_losses['CRPS_diastole'] = CRPS_diastole

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


def log_loss(y, t, eps=1e-7):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.mean(t * np.log(y) + (1-t) * np.log(1-y), axis=(1,))
    return loss




class WeightedLogLossObjective(TargetVarDictObjective):
    def __init__(self, input_layers, *args, **kwargs):
        super(WeightedLogLossObjective, self).__init__(input_layers, *args, **kwargs)
        self.input_systole  = input_layers["systole:onehot"]
        self.input_diastole = input_layers["diastole:onehot"]

        self.target_vars["systole"]  = T.fmatrix("systole_target")
        self.target_vars["diastole"] = T.fmatrix("diastole_target")
        self.target_vars["systole:onehot"]  = T.fmatrix("systole_target_onehot")
        self.target_vars["diastole:onehot"] = T.fmatrix("diastole_target_onehot")
        self.target_vars["systole:class_weight"]  = T.fmatrix("systole_target_weights")
        self.target_vars["diastole:class_weight"] = T.fmatrix("diastole_target_weights")

    def get_loss(self, *args, **kwargs):
        network_systole  = lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs)
        network_diastole = lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs)

        systole_target = self.target_vars["systole:onehot"]
        diastole_target = self.target_vars["diastole:onehot"]
        systole_weights = self.target_vars["systole:class_weight"]
        diastole_weights = self.target_vars["diastole:class_weight"]

        if "average" in kwargs and not kwargs["average"]:
            ll = 0.5 * weighted_log_loss(network_systole, systole_target, weights=systole_weights) + \
                 0.5 * weighted_log_loss(network_diastole, diastole_target, weights=diastole_weights)
            return ll

        ll = 0.5 * T.mean(weighted_log_loss(network_systole, systole_target, weights=systole_weights),  axis = (0,)) + \
             0.5 * T.mean(weighted_log_loss(network_diastole, diastole_target, weights=diastole_weights), axis = (0,))
        return ll + self.penalty

    def get_kaggle_loss(self, validation=False, average=True, *args, **kwargs):
        if not validation:
            return theano.shared([-1])

        network_systole  = T.clip(T.extra_ops.cumsum(lasagne.layers.helper.get_output(self.input_systole, *args, **kwargs), axis=1), 0.0, 1.0).astype('float32')
        network_diastole = T.clip(T.extra_ops.cumsum(lasagne.layers.helper.get_output(self.input_diastole, *args, **kwargs), axis=1), 0.0, 1.0).astype('float32')

        systole_target = self.target_vars["systole"].astype('float32')
        diastole_target = self.target_vars["diastole"].astype('float32')

        if not average:
            CRPS = T.mean((network_systole - systole_target)**2 + (network_diastole - diastole_target)**2, axis = 1)/2
            return CRPS
        else:
            CRPS = (T.mean((network_systole - systole_target)**2,  axis = (0,1)) +
                    T.mean((network_diastole - diastole_target)**2, axis = (0,1)) )/2
            theano_printer.print_me_this("CRPS", CRPS)
            return CRPS


def weighted_log_loss(y, t, weights, eps=1e-7):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.mean(weights * (t * np.log(y) + (1-t) * np.log(1-y)), axis=(1,))
    return loss



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

