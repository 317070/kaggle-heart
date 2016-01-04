import numpy as np
import re
from configuration import config
import scipy

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


