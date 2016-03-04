import numpy as np
from scipy.fftpack import fftn, ifftn
from skimage.draw import ellipse
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle
from skimage.util import img_as_ubyte

def extract_roi(data, pixel_spacing, minradius_mm=25, maxradius_mm=45, kernel_width=5, center_margin=8, num_peaks=10,
                num_circles=20, radstep=2):
    """
    Returns center and radii of ROI region in (i,j) format
    """
    # radius of the smallest and largest circles in mm estimated from the train set
    # convert to pixel counts
    minradius = int(minradius_mm / pixel_spacing)
    maxradius = int(maxradius_mm / pixel_spacing)

    ximagesize = data[0]['data'].shape[1]
    yimagesize = data[0]['data'].shape[2]

    xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
    ysurface = np.tile(range(yimagesize), (ximagesize, 1))
    lsurface = np.zeros((ximagesize, yimagesize))

    allcenters = []
    allaccums = []
    allradii = []

    for dslice in data:
        ff1 = fftn(dslice['data'])
        fh = np.absolute(ifftn(ff1[1, :, :]))
        fh[fh < 0.1 * np.max(fh)] = 0.0
        image = 1. * fh / np.max(fh)

        # find hough circles and detect two radii
        edges = canny(image, sigma=3)
        hough_radii = np.arange(minradius, maxradius, radstep)
        hough_res = hough_circle(edges, hough_radii)

        if hough_res.any():
            centers = []
            accums = []
            radii = []

            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract num_peaks circles
                peaks = peak_local_max(h, num_peaks=num_peaks)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius] * num_peaks)

            # Keep the most prominent num_circles circles
            sorted_circles_idxs = np.argsort(accums)[::-1][:num_circles]

            for idx in sorted_circles_idxs:
                center_x, center_y = centers[idx]
                allcenters.append(centers[idx])
                allradii.append(radii[idx])
                allaccums.append(accums[idx])
                brightness = accums[idx]
                lsurface = lsurface + brightness * np.exp(
                    -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

    lsurface = lsurface / lsurface.max()

    # select most likely ROI center
    roi_center = np.unravel_index(lsurface.argmax(), lsurface.shape)

    # determine ROI radius
    roi_x_radius = 0
    roi_y_radius = 0
    for idx in range(len(allcenters)):
        xshift = np.abs(allcenters[idx][0] - roi_center[0])
        yshift = np.abs(allcenters[idx][1] - roi_center[1])
        if (xshift <= center_margin) & (yshift <= center_margin):
            roi_x_radius = np.max((roi_x_radius, allradii[idx] + xshift))
            roi_y_radius = np.max((roi_y_radius, allradii[idx] + yshift))

    if roi_x_radius > 0 and roi_y_radius > 0:
        roi_radii = roi_x_radius, roi_y_radius
    else:
        roi_radii = None

    ROImask = np.zeros_like(lsurface)
    [rr, cc] = ellipse(roi_center[0], roi_center[1], roi_radii[0], roi_radii[1])
    ROImask[rr, cc] = 1.

    return lsurface, ROImask, roi_center



def extract_roi_joni(data, maxradius, minradius, kernel_width=5, center_margin=8, num_peaks=10, num_circles=20,
                     upscale=1., radstep=2):
    ximagesize = data[0]['data'].shape[1]
    yimagesize = data[0]['data'].shape[2]

    print 'min,max', minradius, maxradius


    xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
    ysurface = np.tile(range(yimagesize), (ximagesize, 1))
    lsurface = np.zeros((ximagesize, yimagesize))

    allcenters = []
    allaccums = []
    allradii = []

    for ddi in data:
        outdata = ddi['data']
        ff1 = fftn(outdata)
        fh = np.absolute(ifftn(ff1[1, :, :]))
        fh[fh < 0.1 * np.max(fh)] = 0.0
        image = 1.*fh / np.max(fh)

        # find hough circles
        edges = canny(image, sigma=3) #, low_threshold=10, high_threshold=50)

        # Detect two radii
        hough_radii = np.arange(minradius, maxradius, radstep)
        hough_res = hough_circle(edges, hough_radii)

        if hough_res.any():
            centers = []
            accums = []
            radii = []

            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract num_peaks circles
                peaks = peak_local_max(h, num_peaks=num_peaks)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius] * num_peaks)

            # Keep the most prominent num_circles circles
            sorted_circle_idxs = np.argsort(accums)[::-1][:num_circles]

            for idx in sorted_circle_idxs:
                center_x, center_y = centers[idx]
                allcenters.append(centers[idx])
                allradii.append(radii[idx])
                allaccums.append(accums[idx])
                brightness = accums[idx]
                lsurface = lsurface + brightness * np.exp(
                    -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

    lsurface = lsurface / lsurface.max()

    # select most likely ROI center
    x_axis, y_axis = np.unravel_index(lsurface.argmax(), lsurface.shape)

    # determine ROI radius
    x_radius = 0
    y_radius = 0
    for idx in range(len(allcenters)):
        xshift = np.abs(allcenters[idx][0] - x_axis)
        yshift = np.abs(allcenters[idx][1] - y_axis)
        if (xshift <= center_margin) & (yshift <= center_margin):
            x_radius = np.max((x_radius, allradii[idx] + xshift))
            y_radius = np.max((y_radius, allradii[idx] + yshift))

    x_radius = upscale * x_radius
    y_radius = upscale * y_radius

    ROImask = np.zeros_like(lsurface)
    [rr, cc] = ellipse(x_axis, y_axis, x_radius, y_radius)
    ROImask[rr, cc] = 1.
    print (x_axis, y_axis), x_radius, y_radius
    return lsurface, ROImask, (x_axis, y_axis)
