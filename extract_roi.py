from skimage.feature import register_translation
import matplotlib.pyplot as plt

import numpy as np
from scipy.fftpack import fftn, ifftn
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import canny
from skimage.draw import ellipse
from skimage.util import img_as_ubyte

from scipy import ndimage as ndi
from sklearn import mixture

from skimage.morphology import watershed, skeletonize, square, disk
from skimage.feature import peak_local_max


def axis_likelyhood_surface(data, kernel_width = 5, center_margin = 8, num_peaks = 10, num_circles = 20, upscale = 1.5, minradius = 15, maxradius = 65, radstep = 2):

    ximagesize=data[0]['data'].shape[1]
    yimagesize=data[0]['data'].shape[2]


    maxradius = max(int(maxradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
    minradius = max(int(minradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
    radstep = max(int(radstep / np.max(data[0]['metadata']['PixelSpacing'])),1)

    xsurface=np.tile(range(ximagesize),(yimagesize,1)).T
    ysurface=np.tile(range(yimagesize),(ximagesize,1))
    lsurface=np.zeros((ximagesize,yimagesize))

    allcenters = []
    allaccums = []
    allradii = []

   
    for ddi in data:
        outdata=ddi['data'].copy()
        ff1 = fftn(outdata)
        fh = np.absolute(ifftn(ff1[1, :, :]))
        fh[fh < 0.1* np.max(fh)] =0.0

        image=img_as_ubyte(fh/np.max(fh))

        # find hough circles
        edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)


        # Detect two radii
        hough_radii = np.arange(minradius, maxradius, radstep)
        hough_res = hough_circle(edges, hough_radii)
       # print str(hough_res)
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

    x_radius = upscale * x_radius
    y_radius = upscale * y_radius

    ROImask = np.zeros_like(lsurface)

    [rr,cc] = ellipse(x_axis, y_axis, x_radius, y_radius)
    dum=(rr < ximagesize) & (rr>=0) & (cc < yimagesize) & (cc>=0)
    rr=rr[dum]
    cc=cc[dum]
    ROImask[rr,cc]=1.

    return lsurface, ROImask, (x_axis, y_axis), (x_radius, y_radius)

def align_images(data):

    numslices=len(data)
    imageshifts = np.zeros((numslices,2))

    # calculate image shifts      imageshifts[idx,:] = imageshifts[idx-1,:] + shift

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


def extract_region(labeled_regions, label):
    component=np.zeros_like(labeled_regions)
    component[labeled_regions == label] = 1
    return component



def select_component_sqdist(labeled_regions, num_labels, center, distmat, opening_param = 3.5, do_diler = True):

    opening_param = int(opening_param)
    mshape = square(3)

    # select closest component
    xmindist = -1
    contains_center = False
    bestcomponent = np.zeros_like(labeled_regions)

    for lidx in range(num_labels):
        reg = extract_region(labeled_regions, lidx+1)
        #smallfilter = ndi.binary_erosion(reg.copy(),structure = mshape,iterations = opening_param)
        smallfilter = reg  # no smallfilter!
        if smallfilter[smallfilter>0].any():   # throw out tiny regions
            if do_diler:
                component=ndi.binary_dilation(reg.copy(),structure = mshape,iterations = opening_param)
            else:
                component = reg
            #dst = np.sum(component*distmat)/np.count_nonzero(component)
            dst = np.min(component*distmat)

            if xmindist < 0:
                xmindist = dst
                bestcomponent = component.copy()
            else:
                if (reg[center] == 1) & (contains_center == False) :
                    contains_center = True
                    xmindist = dst
                    bestcomponent = component.copy()
                else:
                    if (reg[center] == 1) & (contains_center == True) & (dst < xmindist):
                        xmindist = dst
                        bestcomponent = component.copy()

    return bestcomponent, labeled_regions


def extract_segmentation_model(blk):
    # uses a single slice to extract segmentation model
    # still to check: does it improve if you use the whole stack?
    block = blk.copy()/1.0
    #rescalefact = np.max(block)
    #block = block / rescalefact
    #lowerb=0.0
    #upperb=1.0
    ##lowerb=1.0*np.min(block[block > 0.05])
    ##upperb=1.0*np.max(block[block < 0.95])

    perc = 1.0*np.percentile(block[block>0], q=[10, 95])
    lowerb, upperb = perc[0], perc[1]
    rescalefact = 1.0
    block[block>0] = np.clip(1. * (block[block>0] - lowerb) / (upperb - lowerb), 0.0, 1.0)

    obs = block[block>0.]
    print "Shape of obs: " + str(obs.shape)
    obs = np.reshape(obs,(obs.shape[0],1))
    print "Shape of obs: " + str(obs.shape)

    g = mixture.GMM(n_components=2)
    g.fit(obs)
    bloodlabel = g.predict(((0.,),(1.,)))

    return g, bloodlabel, lowerb*rescalefact, upperb*rescalefact


def add_cross_to_image(im, xx, yy):
    image = color.gray2rgb(im.copy())
    for h in range(-4, 5, 1):
        image[xx,yy+h] = (0, 1, 0)

    for v in range(-4, 5, 1):
        image[xx+v,yy] = (0, 1, 0)
    return image

def generate_components(binary):
    # TODO:
    #        - extract acceptable disconnected components, split up connected regions
    #        - find a way to deal with small regions, since you can't throw them out in bottom slices!
    #          (probably keep them and let the selection take care of them)
    pass

def extract_optimal_components(binary_sequence):
    # TODO:
    #       component selection, but use accross sequence information to determine best one
    #       possibly have voting rounds:
    #          - ROI center based only in first round (ranking and/or preselection)
    #          - possibly: use different weights for regions that do or do not contain center
    #          - count 'selectedness' accross stacks and use that for mediating
    pass



def segment_sequence(seq_block, RIOmask = None, segmodel = None, opening_param = 2):

    # generate stack of binary images
    if not(segmodel):
        segmodel = extract_segmentation_model(seq_block)  # segmodel = (g, bloodlabel, lowerb, upperb)

    g, bloodlabel, lowerb, upperb = segmodel
    slice_zdim = seq_block.shape[0]

    labelblock = np.zeros_like(seq_block)
    preprocessblock = np.zeros_like(seq_block)

    for idx in range(slice_zdim):
        if RIOmask:
            patch= seq_block[idx] * RIOmask
        patch[patch>0] = np.clip(1. * (patch[patch>0] - lowerb) / (upperb - lowerb), 0.0, 1.0)

        #patch = ndi.grey_opening(patch, structure = disk(opening_param))
        preprocessblock[idx] = patch.copy()

        dum = patch.flatten()
        dum = np.reshape(dum,(dum.shape[0],1))
        thresh =g.predict(dum)
        thresh = np.reshape(thresh.T,patch.shape)
        labelblock[idx][thresh == bloodlabel[1]] = 1
        labelblock[idx], num_labels = ndi.label(labelblock[idx])

    return labelblock, segmodel, preprocessblock

def best_component(seq_block, center, distmat = None):

    # select closest component

    for sidx in range(seq_block.shape[0]):
        num_labels = np.max(seq_block[sidx])
        region_scores = np.zeros(num_labels)
        xmindist = -1
        contains_center = False
        bestcomponent = np.zeros_like(seq_block)

        for lidx in range(num_labels):
            component = extract_region(seq_block[sidx], lidx+1)
            #dst = np.sum(component*distmat)/np.count_nonzero(component)
            dst = np.min(component*distmat)

            if xmindist < 0:  # first component
                xmindist = dst
            else:
                if (component[center] == 1):
                    if (contains_center == False) :
                        contains_center = True
                        xmindist = dst
                    else:
                        if (dst < xmindist):
                            xmindist = dst



    return bestcomponent, labeled_regions

def breakup_region(component):
    distance = ndi.distance_transform_edt(component)
    skel = skeletonize(component)
    skeldist = distance*skel
    local_maxi = peak_local_max(skeldist, indices=False, footprint=disk(10))
    local_maxi=ndi.binary_closing(local_maxi,structure = disk(4),iterations = 2)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=component)
    return(labels)

def breakup_segmented_sequence(seq_block):

    for idx in range(seq_block.shape[0]):
        numlabels = np.max(seq_block[idx])
        newnumlabels = numlabels
