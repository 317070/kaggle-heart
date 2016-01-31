import numpy as np
import re
from configuration import config
import scipy
import utils

def postprocess(network_outputs_dict):
    """
    convert the network outputs, to the desired kaggle outputs
    """
    kaggle_systoles, kaggle_diastoles = None, None
    if "systole" in network_outputs_dict:
        kaggle_systoles = network_outputs_dict["systole:onehot"]
    if "diastole" in network_outputs_dict:
        kaggle_diastoles = network_outputs_dict["diastole:onehot"]
    if kaggle_systoles is None or kaggle_diastoles is None:
        raise "This is the wrong postprocessing for this model"
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
        raise "This is the wrong postprocessing for this model"
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
        raise "This is the wrong postprocessing for this model"
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


