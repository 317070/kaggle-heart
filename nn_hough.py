
import numpy as np


def _multi_logical_or(*args):
    res = args[0]
    for arg in args[1:]:
       res = np.logical_or(res, arg)
    return res 


def _create_hough_filters(size, rads, margin=np.sqrt(2)/2):
    """Creates the hough filters.

    Args:
      size: width and height of the filter.
      rads: radii of the filter circles.
      margin: distance a pixel can be from the intended radius for it to be
        concidered part of the circle.

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
    
    return circles
    

if __name__ == '__main__':
    _create_hough_filters(10, [4, 5])