"""Library implementing the Hough transform in Theano and Lasagne.
"""


import numpy as np

import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer


def _multi_logical_or(*args):
    res = args[0]
    for arg in args[1:]:
       res = np.logical_or(res, arg)
    return res 


def _create_hough_filters(size, rads, normalise=True):
    """Creates the hough filters.

    Args:
      size: width and height of the filter.
      rads: radii of the filter circles.
      normalise: if True (default), all filters' DC gain is set to 1. Otherwise,
          larger circles would end up having larger  

    Returns:
      3 dimentional Numpy Array containing the filters. The dimensions are:
      channel (i.e. circle radius), width, and height.
    """
    size = float(size)
    y, x = np.mgrid[0:size, 0:size] - size/2+.5
    dists_sq = x ** 2 + y ** 2
    dists_tr_sq = (x + .5) ** 2 + (y + .5) ** 2
    dists_br_sq = (x + .5) ** 2 + (y - .5) ** 2
    dists_bl_sq = (x - .5) ** 2 + (y - .5) ** 2
    dists_tl_sq = (x - .5) ** 2 + (y + .5) ** 2
    crnr_dists = (dists_tr_sq, dists_br_sq, dists_bl_sq, dists_tl_sq)

    circles = np.zeros((len(rads), size, size))
    for idx, rad in enumerate(rads):
        rad_sq = rad ** 2
        any_in_circle = _multi_logical_or(*[cd < rad_sq for cd in crnr_dists])
        any_out_circle = _multi_logical_or(*[cd > rad_sq for cd in crnr_dists])
        circles[idx, :, :] = np.logical_and(any_in_circle, any_out_circle)
        
    if normalise:
        # Alternative: divide by radius (DC gain will not be 1, but
        # approximately equal for all filters)
        # circles /= rads[:, np.newaxis, np.newaxis]
        circles /= circles.sum(axis=2).sum(axis=1)[:, np.newaxis, np.newaxis]

    return circles


class HoughDNNLayer(Conv2DDNNLayer):
    def __init__(self, incoming, radii, normalise=True, 
                 stride=(1, 1), pad=0, **kwargs):
        # Use predetermined Hough filters
        W = _create_hough_filters(2*np.max(radii)+1, radii, normalise=normalise)
        # Transform to correct shape and dtype 
        W = W[:, np.newaxis, :, :].astype('float32')

        # remove biasses and nonlinearities
        b = None
        untie_biases = False
        nonlinearity = None
        flip_filters = True  # doesn't matter
        super(Conv2DDNNLayer, self).__init__(incoming, W.shape[0],
                                             W.shape[-2:], stride, pad,
                                             untie_biases, W, b, nonlinearity,
                                             flip_filters, n=2, **kwargs)
        # Remove trainable tag for W
        self.params[self.W] = self.params[self.W].difference(set(['trainable']))
