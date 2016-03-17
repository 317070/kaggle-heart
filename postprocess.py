"""Library implementing different post-process functions.

Post-process functions are functions that transform the network output to
a suitable format (i.e. a series of 600, monotonously increasing numbers).
"""
import re

import numpy as np
import scipy

import utils

from configuration import config


def make_monotone_distribution(distribution):
    if distribution.ndim==1:
        for j in xrange(len(distribution)-1):
            if not distribution[j] <= distribution[j+1]:
                distribution[j+1] = distribution[j]
        distribution = np.clip(distribution, 0.0, 1.0)
        return distribution
    else:
        return np.apply_along_axis(make_monotone_distribution, axis=-1, arr=distribution)


def make_monotone_distribution_fast(distributions):
    return utils.pdf_to_cdf(np.clip(utils.cdf_to_pdf(distributions), 0.0, 1.0))


def test_if_valid_distribution(distribution):
    if not np.isfinite(distribution).all():
        raise Exception("There is a non-finite number in there")

    for j in xrange(len(distribution)):
        if not 0.0<=distribution[j]<=1.0:
            raise Exception("There is a number smaller than 0 or bigger than 1: %.18f" % distribution[j])

    for j in xrange(len(distribution)-1):
        if not distribution[j] <= distribution[j+1]:
            raise Exception("This distribution is non-monotone: %.18f > %.18f" % (distribution[j], distribution[j+1]))


def postprocess(network_outputs_dict):
    """
    convert the network outputs, to the desired kaggle outputs
    """
    kaggle_systoles, kaggle_diastoles = None, None
    if "systole" in network_outputs_dict:
        kaggle_systoles = network_outputs_dict["systole"]
    if "diastole" in network_outputs_dict:
        kaggle_diastoles = network_outputs_dict["diastole"]
    if kaggle_systoles is None or kaggle_diastoles is None:
        raise Exception("This is the wrong postprocessing for this model")

    mmd = make_monotone_distribution_fast
    return mmd(kaggle_systoles), mmd(kaggle_diastoles)


def postprocess_onehot(network_outputs_dict):
    """
    convert the network outputs, to the desired kaggle outputs
    """
    kaggle_systoles, kaggle_diastoles = None, None
    if "systole:onehot" in network_outputs_dict:
        kaggle_systoles = np.clip(np.cumsum(network_outputs_dict["systole:onehot"], axis=1), 0.0, 1.0)
    if "diastole:onehot" in network_outputs_dict:
        kaggle_diastoles = np.clip(np.cumsum(network_outputs_dict["diastole:onehot"], axis=1), 0.0, 1.0)
    if kaggle_systoles is None or kaggle_diastoles is None:
        raise Exception("This is the wrong postprocessing for this model")
    return kaggle_systoles, kaggle_diastoles


def postprocess_value(network_outputs_dict):
    """Convert the network outputs to a Gaussian distribution.

    The network should have the outputs:
    - systole:value
    - diastole:value
    - systole:sigma (optional, default=0)
    - diastole:sigma (optional, default=0)
    """
    kaggle_systoles, kaggle_diastoles = None, None
    if "systole:value" in network_outputs_dict:
        mu = network_outputs_dict["systole:value"][:,0]
        if "systole:sigma" in network_outputs_dict:
            sigma = network_outputs_dict["systole:sigma"][:,0]
        else:
            sigma = np.zeros_like(mu)
        kaggle_systoles = utils.numpy_mu_sigma_erf(mu, sigma)
    if "diastole:value" in network_outputs_dict:
        mu = network_outputs_dict["diastole:value"][:,0]
        if "diastole:sigma" in network_outputs_dict:
            sigma = network_outputs_dict["diastole:sigma"][:,0]
        else:
            sigma = np.zeros_like(mu)
        kaggle_diastoles = utils.numpy_mu_sigma_erf(mu, sigma)
    if kaggle_systoles is None or kaggle_diastoles is None:
        raise Exception("This is the wrong postprocessing for this model")
    return kaggle_systoles, kaggle_diastoles



def upsample_segmentation(original, output_shape, order=1):
    """
    upsample a float segmentation image last dimensions until they match
    the output_shape
    (by bilinear interpolating)
    :param original:
    :return:
    """
    #print original.shape
    z = []
    for i in xrange(original.ndim):
        if len(output_shape) - original.ndim + i < 0:
            z.append(1)
        else:
            z.append(output_shape[len(output_shape) - original.ndim + i] / original.shape[i])
    #print z
    result = scipy.ndimage.interpolation.zoom(original, zoom=z, order=order)
    #print result.shape
    return result


