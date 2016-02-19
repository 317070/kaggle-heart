import numpy as np
from scipy.fftpack import fftn, ifftn
from skimage.transform import hough_circle
from skimage.transform import SimilarityTransform
from skimage.transform import warp

from skimage.feature import peak_local_max, canny
from skimage.draw import ellipse
from skimage.util import img_as_ubyte
from skimage.feature import register_translation
import matplotlib.pyplot as plt


def axis_likelyhood_surface(data, kernel_width = 5, center_margin = 8, num_peaks = 10, num_circles = 20, upscale = 1.5, minradius = 20, maxradius = 60, radstep = 2):

    ximagesize=data[0]['data'].shape[1]
    yimagesize=data[0]['data'].shape[2]

    xsurface=np.tile(range(ximagesize),(yimagesize,1)).T
    ysurface=np.tile(range(yimagesize),(ximagesize,1))
    lsurface=np.zeros((ximagesize,yimagesize))

    allcenters = []
    allaccums = []
    allradii = []


    for ddi in data:
        outdata=ddi['data']
        ff1 = fftn(outdata)
        fh = np.absolute(ifftn(ff1[1, :, :]))
        fh[fh < 0.1* np.max(fh)] =0.0
        image=img_as_ubyte(fh/np.max(fh))

        # find hough circles
        edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

        # Detect two radii
        hough_radii = np.arange(minradius, maxradius, radstep)
        hough_res = hough_circle(edges, hough_radii)

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
        sorted=np.argsort(accums)[::-1][:num_circles]

        for idx in sorted:
            center_x, center_y = centers[idx]
            allcenters.append(centers[idx])
            allradii.append(radii[idx])
            allaccums.append(accums[idx])
            brightness=accums[idx]
            lsurface = lsurface + brightness* np.exp(-((xsurface-center_x)**2 + (ysurface-center_y)**2)/kernel_width**2)

    lsurface=lsurface/lsurface.max()

    # select most likely ROI center
    x_axis, y_axis = np.unravel_index(lsurface.argmax(), lsurface.shape)

    # determine ROI radius
    x_radius = 0
    y_radius = 0
    for idx in range(len(allcenters)):
        xshift=np.abs(allcenters[idx][0]-x_axis)
        yshift=np.abs(allcenters[idx][1]-y_axis)
        if (xshift <= center_margin) & (yshift <= center_margin):
            x_radius = np.max((x_radius,allradii[idx]+xshift))
            y_radius = np.max((y_radius,allradii[idx]+yshift))

    #print x_axis, y_axis, x_radius, y_radius
    #print allcenters

    x_radius = upscale * x_radius
    y_radius = upscale * y_radius

    ROImask = np.zeros_like(lsurface)
    [rr,cc] = ellipse(x_axis, y_axis, x_radius, y_radius)
    ROImask[rr,cc]=1.
    #plt.figure()
    #plt.imshow(ROImask, cmap='gist_gray_r', vmin=0., vmax=1.)
    #plt.show()

    return lsurface, ROImask, (x_axis, y_axis), (x_radius, y_radius)

def align_images(data):

    numslices=len(data)
    imageshifts = np.zeros((numslices,2))

    # calculate image shifts
    for idx in range(numslices):
        if idx == 0:
            pass
        else:
            image = np.mean(data[idx-1]['data'],0)
            offset_image = np.mean(data[idx]['data'],0)

            ## shifts in pixel precision for speed
            shift, error, diffphase = register_translation(image, offset_image)
            imageshifts[idx,:] = imageshifts[idx-1,:] + shift

    # apply image shifts
    for idx in range(numslices):
        non = lambda s: s if s<0 else None
        mom = lambda s: max(0,s)
        padded = np.zeros_like(data[idx]['data'])
        oy, ox = imageshifts[idx,:]
        padded[:,mom(oy):non(oy), mom(ox):non(ox)] = data[idx]['data'][:,mom(-oy):non(-oy), mom(-ox):non(-ox)]
        data[idx]['data']=padded.copy()
        #tform=SimilarityTransform(translation = imageshifts[idx,:])
        #for idx2 in range(data[idx]['data'].shape[0]):
        #    tformed = warp(data[idx]['data'][idx2,:,:], inverse_map = tform)
        #    data[idx]['data'][idx2,:,:]= tformed

    return data

def sort_images(data):
    numslices=len(data)
    positions=np.zeros((numslices,))
    for idx in range(numslices):
        positions[idx] = data[idx]['metadata']['SliceLocation']
    newdata=[x for y, x in sorted(zip(positions.tolist(), data), key=lambda dd: dd[0], reverse=True)]
    return newdata