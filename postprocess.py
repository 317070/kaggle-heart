import numpy as np
import re
from configuration import config
import scipy


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


