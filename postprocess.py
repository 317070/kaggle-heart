import numpy as np
import re
from configuration import config
import scipy
import utils


def make_monotone_distribution(distribution):
    for i in xrange(len(distribution)):
        for j in xrange(len(distribution[0])-1):
            if not distribution[i,j] <= distribution[i,j+1]:
                distribution[i,j+1] = distribution[i,j]
    distribution = np.clip(distribution, 0.0, 1.0)
    return distribution

def test_if_valid_distribution(distribution):
    print distribution.shape
    if not np.isfinite(distribution).all():
        raise Exception("There is a non-finite numer in there")
    """
    for i in xrange(len(distribution)):
        for j in xrange(len(distribution[0])):
            if not 0.0<=distribution[i,j]<=1.0:
                raise Exception("There is a number smaller than 0 or bigger than 1: %.18f" % distribution[i,j])
    """
    for i in xrange(len(distribution)):
        for j in xrange(len(distribution[0])-1):
            if not distribution[i,j] <= distribution[i,j+1]:
                print distribution[i]
                print distribution.shape
                print distribution.dtype
                raise Exception("This distribution is non-monotone: %.18f > %.18f" % (distribution[i,j], distribution[i,j+1]))


def postprocess(network_outputs_dict):
    """
    convert the network outputs, to the desired kaggle outputs
    """
    kaggle_systoles, kaggle_diastoles = None, None
    if "systole" in network_outputs_dict:
        kaggle_systoles = make_monotone_distribution(network_outputs_dict["systole"])
        #kaggle_systoles = np.clip(network_outputs_dict["systole"], 0.0, 1.0)
    if "diastole" in network_outputs_dict:
        kaggle_diastoles = make_monotone_distribution(network_outputs_dict["diastole"])
        #kaggle_diastoles = np.clip(network_outputs_dict["diastole"], 0.0, 1.0)
    if kaggle_systoles is None or kaggle_diastoles is None:
        raise Exception("This is the wrong postprocessing for this model")

    try:
        test_if_valid_distribution(kaggle_systoles)
        test_if_valid_distribution(kaggle_diastoles)
    except:
        print "These distributions are not distributions"

    kaggle_systoles  = make_monotone_distribution(kaggle_systoles)
    kaggle_diastoles  = make_monotone_distribution(kaggle_diastoles)

    return kaggle_systoles, kaggle_diastoles


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
    """
    convert the network outputs, to the desired kaggle outputs
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
        print mu

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


