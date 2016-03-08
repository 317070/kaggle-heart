import cPickle as pickle
from scipy.special import erf
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from skimage.feature import register_translation
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from scipy.fftpack import fftn, ifftn
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import canny
from skimage.draw import ellipse
from skimage.util import img_as_ubyte

from scipy import ndimage as ndi
from sklearn import mixture

from skimage.morphology import watershed, skeletonize, square, disk, convex_hull_image
from skimage.feature import peak_local_max

from scipy.signal import butter, lfilter, freqz
from image_transform import fast_warp
from paths import TRAIN_PATIENT_IDS
from util_scripts.test_simul_ch import get_chan_transformations, clean_metadata
from util_scripts.test_simul_ch import _enhance_metadata
from util_scripts.test_slice_locationing import slice_location_finder


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
        #image = fh/np.max(fh)

        # find hough circles
        edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
        #edges = canny(image)

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

    #x_radius = 1.5*maxradius
    #y_radius = 1.5*maxradius

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


def extract_segmentation_model(blk, do_plot = False):
    # uses a single slice to extract segmentation model
    # still to check: does it improve if you use the whole stack?
    block = 1.0*blk.copy()
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
    obs = np.reshape(obs,(obs.shape[0],1))

    if do_plot:
        plt.figure()
        plt.hist(obs.ravel())
        plt.show()

    g = mixture.GMM(n_components=2)
    g.fit(obs)
    bloodlabel = g.predict(((0.,),(1.,)))

    # find threshold up to 1e-5 accuracy
    accuracy = 1.0e-5
    xmax= 1.
    xmin = 0.
    while (xmax - xmin) > accuracy:
        newp = (xmax + xmin)/2.
        newl = g.predict([newp])
        if newl == bloodlabel[0]:
            xmin = newp
        else:
            xmax = newp
        #print "[" + str(xmin) + "," + str(xmax) + "]"

    threshold = (xmax + xmin)/2.
    #print "GMM Threshold = " + str(threshold)

    return threshold, bloodlabel, lowerb*rescalefact, upperb*rescalefact


def add_cross_to_image(im, xx, yy):
    image = color.gray2rgb(im.copy())
    for h in range(-4, 5, 1):
        image[xx,yy+h] = (0, 1, 0)

    for v in range(-4, 5, 1):
        image[xx+v,yy] = (0, 1, 0)
    return image


def generate_fft_image(sequence):
    ff1 = fftn(sequence)
    fh = np.absolute(ifftn(ff1[1, :, :]))
    fh[fh < 0.1* np.max(fh)] =0.0
    #image=img_as_ubyte(fh/np.max(fh))
    return fh


def segment_sequence(seq_block, segmodel = None):

    # generate stack of binary images:
    #
    # seq_block = 3d of array num_time_steps * image
    #
    # seg_model = (g, bloodlabel, lowerb, upperb)
    #             if segmodel is not specified, it will be determined below, i.e. on the current block
    #             pass on a segmodel is you want to reuse one that was determined on another block
    #             (this assumes that image contrast and lighness are similar accross all sequences,
    #              but when slices occur that contain little or no blood volume, this is usually a better approach)


    if segmodel == None:
        segmodel = extract_segmentation_model(seq_block)  # segmodel = (g, bloodlabel, lowerb, upperb)

    threshold, bloodlabel, lowerb, upperb = segmodel
    slice_zdim = seq_block.shape[0]

    labelblock = np.zeros_like(seq_block)

    for idx in range(slice_zdim):
        patch = 1.0*seq_block[idx].copy()   # make sure the numbers are floats
        patch[patch>0] = np.clip(1. * (patch[patch>0] - lowerb) / (upperb - lowerb), 0.0, 1.0)

        labelblock[idx][patch>threshold] = 1
        #dum = patch.flatten()
        #dum = np.reshape(dum,(dum.shape[0],1))
        #thresh =g.predict(dum)
        #thresh = np.reshape(thresh.T,patch.shape)
        #labelblock[idx][thresh == bloodlabel[1]] = 1
        ##labelblock[idx], num_labels = ndi.label(labelblock[idx])

    return labelblock, segmodel


def extract_binary_regions(sequence, opening_param = 3, mshape = ((0,1,0),(1,1,1),(0,1,0))):

    labelblock = np.zeros_like(sequence)

    for idx in range(sequence.shape[0]):
        labelblock[idx] = sequence[idx].copy()
        labelblock[idx] = ndi.binary_opening(labelblock[idx], iterations = opening_param, structure = mshape)
        labelblock[idx], num_labels = ndi.label(labelblock[idx])

    return labelblock, opening_param, mshape


def split_up_binary_regions(seq_block, opening_param = 3, mshape = ((0,1,0),(1,1,1),(0,1,0)), min_size = 20):

    # for each region:
    #                  dilate, relabel, create stack with eroded single regions
    #                  delete overlapping pixels, recombine, assigning unique labels

    zdim, xdim, ydim = seq_block.shape
    splitblock = np.zeros_like(seq_block)

    for sidx in range(zdim):
        num_labels = int(np.max(seq_block[sidx]))
        if num_labels <= 1:
            continue
        max_label = num_labels
        for lidx in range(num_labels):
            comp = extract_region(seq_block[sidx], lidx+1)
            if np.sum(comp) < min_size:
                continue
            component = ndi.binary_erosion(comp,structure = mshape,iterations = opening_param)
            component, numnewcomponents = ndi.label(component)
            if numnewcomponents == 1:  # component is not split up by erosion
                #component[component>0] = 1
                #component = ndi.binary_dilation(component,structure = mshape,iterations = opening_param)
                splitblock[sidx][comp > 0] = int(lidx+1)
            else:
                compstack = np.zeros((numnewcomponents,xdim,ydim))
                for cidx in range(numnewcomponents):
                    compstack[cidx] = ndi.binary_dilation(extract_region(component, cidx+1),structure = mshape,iterations = opening_param)
                    compstack[cidx] = ndi.binary_dilation(compstack[cidx],structure = ((0,1,0),(1,1,1),(0,1,0)),iterations = 2)
                overlapmask = np.sum(compstack, axis = 0)
                comp[overlapmask > 1] = 0
                comp, numnewcomponents2 = ndi.label(comp)
                #if numnewcomponents != numnewcomponents2:
                    #print "Lost something on the way: C1 = " + str(numnewcomponents) + ", C2 = " + str(numnewcomponents2)

                for cidx in range(numnewcomponents2):
                    component = extract_region(comp, cidx+1)
                    if np.sum(component) < min_size:
                        continue
                    #compstack[cidx] = compstack[cidx] * overlapmask
                    #if np.sum(compstack[cidx]) > 0:
                    max_label = max_label + 1
                    #compstack[cidx] = compstack[cidx] * max_label
                    splitblock[sidx] = splitblock[sidx] + component.astype(int) * int(max_label)

                #splitblock[sidx] = splitblock[sidx] + np.sum(compstack,axis = 0).astype(int)

        # relabel components to unique labels
        all_labels = np.sort(np.unique(splitblock[sidx]))
        dum = np.zeros_like(splitblock[sidx])
        l_idx = 1
        for label in all_labels:
            if label >0:
                dum[splitblock[sidx]==label] = l_idx
                l_idx = l_idx + 1

    return splitblock

def best_component(seq_block, center, distmat = None, \
                   use_kernel = True, kernel_width = 20, do_plot = False, min_size = 20):

    # select closest component
    # seq_block = sequence of labeled regions
    # center = ROI center
    # distmat = matrix with squared distance from center (more efficient if calculated only once)
    # use kernel:
    #             if True: select based on region overlap AND distance from center in stage 2,
    #             otherwise only use overlap from first pass
    #             Note: always use kernel when first pass returns nothing

    zdim, xdim, ydim = seq_block.shape
    x_center, y_center = center
    bestindices = np.zeros(zdim)

    if distmat == None:
        xsurface=np.tile(range(xdim),(ydim,1)).T
        ysurface=np.tile(range(ydim),(xdim,1))
        distmat = (xsurface-x_center)**2 + (ysurface-y_center)**2
        distkernel = np.exp(-((xsurface-x_center)**2 + (ysurface-y_center)**2)/kernel_width**2)
    else:
        distkernel = np.exp(-distmat/kernel_width**2)

    contains_center = np.zeros(zdim)
    region_selected = np.zeros(zdim)
    bestcomponent = np.zeros_like(seq_block)

    # stage 1: select regions containing ROI

    for sidx in range(zdim):
        #num_labels = int(np.max(seq_block[sidx]))
        center_label = seq_block[sidx][center]
        if center_label > 0:
            contains_center[sidx] = 1
            region_selected[sidx] = 1
            bestcomponent[sidx] = extract_region(seq_block[sidx], center_label)
            bestindices[sidx] = center_label
        #for lidx in range(num_labels):
        #    component = extract_region(seq_block[sidx], lidx+1)
        #    if (component[center] == 1):
        #        contains_center[sidx] = 1
        #        region_selected[sidx] = 1
        #        bestcomponent[sidx] = component

    # stage 2: create likelyhood surface: based on overlap only or + centered kernel

    if np.sum(region_selected) > 0: # case where at least one region is selected
        #print "Stage 2"

        numselected = region_selected.sum()
        if numselected == zdim: # if regions have been selected for all slices: stop here
            print "Components have been selected for all time steps in stage 1"
            return bestcomponent, bestindices

        print "Components have been selected for " + str(numselected) + "/" + str(zdim) + " time steps in stage 1"
        lsurface = np.sum(1.0*bestcomponent, axis = 0)
        lsurface = lsurface / (1.0*numselected)
        if use_kernel:
            lsurface *= distkernel
    else:  # Return  all-"-1" masks to make this detectable"
        print "No components contain ROI!!"
        return - np.ones_like(seq_block).astype(int), bestindices
        #lsurface = distkernel

    # stage 3: select regions for other slices depending on likelyhood surface
    # selection criterion? max mean of masked likelyhood surface? favours small regions!
    # excluding nonoverlapping pixels?
    # max sum of likelyhood surface? (favours regions with large absolute overlap)

    #print "Stage 3"

    for sidx in range(zdim):

        if region_selected[sidx] == 0 :

            num_labels = int(np.max(seq_block[sidx]))
            if num_labels == 0:
                continue
            regionscore_sum = np.zeros(num_labels)
            regionscore_mean = np.zeros(num_labels)
            regionscore_max = np.zeros(num_labels)

            for lidx in range(num_labels):
                component = extract_region(seq_block[sidx], lidx+1)

                if np.sum(component) < min_size:
                    continue
                npixels = component.sum()
                wcomponent = 1.0 * component * lsurface

                regionscore_sum[lidx] = np.sum(wcomponent[wcomponent > 0])
                regionscore_mean[lidx] = regionscore_sum[lidx]/npixels
                regionscore_max[lidx] = np.max(wcomponent)

                #if regionscore_sum[lidx]>0:
                #    bestcomponent[sidx] = bestcomponent[sidx] + label * component.astype(int)
                #    label = label + 1


            best_idx = np.argmax(regionscore_mean)
            if(regionscore_mean[best_idx]) == 0:
                print "Stage 3: no component selected for time step " + str(sidx)
            else:
                bestcomponent[sidx] = extract_region(seq_block[sidx], best_idx + 1)
                bestindices[sidx] = best_idx + 1

        #if do_plot:
        #    plt.figure()
        #    plt.subplot(221)
        #    plt.imshow(seq_block[sidx])
        #    plt.subplot(222)
        #    plt.imshow(lsurface)
        #    plt.subplot(223)
        #    plt.imshow(bestcomponent[sidx])
        #    plt.subplot(224)
        #    dum = bestcomponent[sidx].copy()
        #    dum[dum > 0] = 1
        #    plt.imshow(dum*lsurface)
        #    plt.show()

    # stage 4 (???): update likelyhood surface, go through the stack again and do final selection

    #print  "Stage 4"

    bestcomponent2 = np.zeros_like(bestcomponent)
    dum = 1.0*bestcomponent.copy()
    dum[dum>0] = 1.0
    lsurface = np.sum(dum, axis = 0)
    core_max = int(zdim/3)
    lsurface[lsurface < core_max] = 0.0
    lsurface = lsurface / core_max

    for sidx in range(zdim):
        npixels = np.sum(bestcomponent[sidx])
        if npixels == 0:
            continue

        wcomponent = (1.0 * bestcomponent[sidx]) * lsurface

        regionscore_sum = np.sum(wcomponent[wcomponent > 0])
        regionscore_mean = regionscore_sum/npixels
        regionscore_max = np.max(wcomponent)

        if(regionscore_mean) == 0:
            print "Stage 4: no component selected for time step " + str(sidx)
            bestindices[sidx] = 0
        else:
            bestcomponent2[sidx] = bestcomponent[sidx].copy()

        if do_plot:
            plt.figure()
            plt.subplot(221)
            plt.imshow(seq_block[sidx])
            plt.subplot(222)
            plt.imshow(bestcomponent[sidx])
            plt.subplot(223)
            plt.imshow(lsurface)
            plt.subplot(224)
            plt.imshow(bestcomponent2[sidx])
            plt.show()

    return bestcomponent2, bestindices

def cut_superfluous_parts(bestregions, regions, bestindices, do_plot = False):

    zdim, xdim, ydim = bestregions.shape
    smask = np.ones((xdim, ydim)).astype(int)

    cutbestregions = bestregions.copy()

    for sidx in range(zdim):
        dum = regions[sidx].copy()
        dum[dum == bestindices[sidx]] = 0
        smask[dum>0] = 0

    for sidx in range(zdim):
        if np.sum(cutbestregions[sidx] > 0):
            dum, numlabels = ndi.label(cutbestregions[sidx] * smask)

            # cut away stray pixels
            maxregion = 0
            for lidx in range(numlabels):
                component = extract_region(dum, lidx + 1)
                npixels = np.sum(component)
                if npixels > maxregion:
                    maxregion = npixels
                    cutbestregions[sidx] = component

        if do_plot:
            plt.figure()
            plt.subplot(131)
            plt.imshow(bestregions[sidx])
            plt.subplot(132)
            plt.imshow(smask)
            plt.subplot(133)
            plt.imshow(cutbestregions[sidx])
            plt.show()

    return cutbestregions

def wrapper_regions(bestregions, opening_param = 3, mshape = ((0,1,0),(1,1,1),(0,1,0)) ):

    zdim, xdim, ydim = bestregions.shape

    wregions = np.zeros_like(bestregions)

    for sidx in range(zdim):
        if np.sum(bestregions[sidx]) > 0:
            wregions[sidx] = convex_hull_image(bestregions[sidx])

    return wregions

def breakup_region(component):
    distance = ndi.distance_transform_edt(component)
    skel = skeletonize(component)
    skeldist = distance*skel
    local_maxi = peak_local_max(skeldist, indices=False, footprint=disk(10))
    local_maxi=ndi.binary_closing(local_maxi,structure = disk(4),iterations = 2)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=component)
    return(labels)


def filter_sequence(seq_block, order = 5, relcutoff = 0.1):

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data, axis = 0)
        return y

    zdim = seq_block.shape[0]
    xdim = seq_block.shape[1]
    ydim = seq_block.shape[2]

    ffdata = np.zeros((3*zdim, xdim, ydim))
    ffdata[:zdim] = seq_block
    ffdata[zdim:2*zdim] = seq_block
    ffdata[2*zdim:3*zdim] = seq_block

    ffdata = butter_lowpass_filter(ffdata, relcutoff, 1.0, order)

    ffdata = ffdata[zdim:2*zdim,:,:]

    return ffdata

def segment_data(data):

    kernel_width = 5
    center_margin = 8
    num_peaks = 10
    num_circles = 10  # 20
    upscale = 1.5
    minradius = 15
    maxradius = 65
    radstep = 2

    segmented = {}

    processdata=sort_images(data)

    numslices = len(processdata)

    if(numslices == 1):
        centerslice = 0
    else:
        centerslice=numslices/2

    segment_block = processdata[centerslice]['data'].copy()  # make sure the numbers are floats
    processdata = data

    zdim, xdim, ydim = segment_block.shape

    lsurface, ROImask, ROIcenter, ROIaxes = axis_likelyhood_surface(processdata, kernel_width = kernel_width,
                                                                    center_margin = center_margin,
                                                                    num_peaks = num_peaks,
                                                                    num_circles = num_circles,
                                                                    upscale = upscale,
                                                                    minradius = minradius,
                                                                    maxradius = maxradius,
                                                                    radstep = radstep)

    #rs_minradius = max(int(minradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
    #rs_maxradius = max(int(maxradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
    #x_center = ROIcenter[0]
    #y_center = ROIcenter[1]
    #x_axis=ROIaxes[0]
    #y_axis = ROIaxes[1]

    # fit GMM  -  I do it only once, but you could also try re-doing it for each slice

    segmodel = extract_segmentation_model(segment_block)

    # apply to slices

    for slice in range(numslices):
        opening_param = 3
        mshape = square(3)

        print "slice nr " + str(slice)
        #if slice < 11:
        #    continue
        #if slice == 7:
        #    print "stop here"

        #sld = filter_sequence(processdata[slice]['data'])
        sld = processdata[slice]['data']

        for idx in range(sld.shape[0]):
            sld[idx] = sld[idx]*ROImask

        binary, sm = segment_sequence(sld,segmodel = segmodel)
        regions, dum1, dum2 = extract_binary_regions(binary)
        regions = split_up_binary_regions(regions, opening_param = 1, mshape = disk(3))#opening_param = 5)
        bestregions, bestindices = best_component(regions, ROIcenter,use_kernel = False, kernel_width = 5, do_plot = False)
        bestregions = cut_superfluous_parts(bestregions, regions, bestindices, do_plot = False)
        bestregions = wrapper_regions(bestregions)

        segmented[slice] = {'roi_center': ROIcenter, 'roi_radii': ROIaxes,'segmask': bestregions.astype(bool)}

    return segmented


def best_component_ch(labeled_image):

    sh = labeled_image.shape[-1]
    selector = labeled_image[:,:,sh/2]
    best_classes = np.array([scipy.stats.mstats.mode(s[s>0]).mode[0] for s in selector])
    bestregions = np.array([extract_region(im, best_class_i) for im, best_class_i in zip(labeled_image, best_classes)])
    return bestregions, best_classes


def numpy_mu_sigma_erf(mu_erf, sigma_erf, eps=1e-7):
    x_axis = np.arange(0, 600, dtype='float32')
    mu_erf = np.tile(mu_erf, (600,))
    sigma_erf = np.tile(sigma_erf, (600,))
    sigma_erf += eps

    x = (x_axis - mu_erf) / (sigma_erf * np.sqrt(2))
    return (erf(x) + 1)/2


if __name__ == "__main__":
    DATA_PATH = "/home/oncilladock/"
    do_plot = False
    labels = pickle.load(open(DATA_PATH+"train.pkl"))

    data_dump = []
    if not os.path.isfile("segmentation.pkl"):
        for patient_id in xrange(1, 701):
            print "Looking for the pickle files..."
            if patient_id<=TRAIN_PATIENT_IDS[1]:
                files = sorted(glob.glob(os.path.expanduser(DATA_PATH+"pkl_train/%d/study/*.pkl" % patient_id)))
            else:
                files = sorted(glob.glob(os.path.expanduser(DATA_PATH+"pkl_validate/%d/study/*.pkl" % patient_id)))

            ch2_file = [f for f in files if "2ch" in f][0]
            if len([f for f in files if "4ch" in f]) > 0:
                has_ch4 = True
                ch4_file = [f for f in files if "4ch" in f][0]
            else:
                has_ch4 = False
                ch4_file = ch2_file

            sax_files = [f for f in files if "sax" in f]
            print "%d sax files" % len(sax_files)

            ch2_metadata = clean_metadata(pickle.load(open(ch2_file))["metadata"][0])
            ch4_metadata = clean_metadata(pickle.load(open(ch4_file))["metadata"][0])

            ch2_data = pickle.load(open(ch2_file))["data"]
            ch4_data = pickle.load(open(ch4_file))["data"]

            metadata_dict = dict()
            for file in files:
                if "sax" in file:
                    all_data = pickle.load(open(file,"r"))
                    metadata_dict[file] = all_data['metadata'][0]
            datadict, sorted_indices, sorted_distances = slice_location_finder(metadata_dict)

            # find top and bottom of my view

            top_point_enhanced_metadata = datadict[sorted_indices[0]]["middle_pixel_position"]
            bottom_point_enhanced_metadata = datadict[sorted_indices[-1]]["middle_pixel_position"]

            top_point_enhanced_metadata = pickle.load(open(sorted_indices[0],"r"))['metadata'][0]
            _enhance_metadata(top_point_enhanced_metadata, patient_id, slice_name = os.path.basename(sorted_indices[0]))

            bottom_point_enhanced_metadata = pickle.load(open(sorted_indices[-1],"r"))['metadata'][0]
            _enhance_metadata(bottom_point_enhanced_metadata, patient_id, slice_name = os.path.basename(sorted_indices[-1]))

            OUTPUT_SIZE = 100

            trf_2ch, trf_4ch = get_chan_transformations(
                ch2_metadata=ch2_metadata,
                ch4_metadata=ch4_metadata if has_ch4 else None,
                top_point_metadata = top_point_enhanced_metadata,
                bottom_point_metadata = bottom_point_enhanced_metadata,
                output_width=OUTPUT_SIZE
                )

            ch4_result = np.array([fast_warp(ch4, trf_4ch, output_shape=(OUTPUT_SIZE, OUTPUT_SIZE)) for ch4 in ch4_data])

            ch2_result = np.array([fast_warp(ch2, trf_2ch, output_shape=(OUTPUT_SIZE, OUTPUT_SIZE)) for ch2 in ch2_data])

            segmodel = extract_segmentation_model(ch2_result)
            ch2_binary, sm = segment_sequence(ch2_result, segmodel = segmodel)

            ch2_regions, dum1, dum2 = extract_binary_regions(ch2_binary)
            #regions = split_up_binary_regions(regions, opening_param = 1, mshape = disk(3))#opening_param = 5)
            ch2_bestregions, ch2_bestindices = best_component_ch(ch2_regions)
            #ch2_bestregions = cut_superfluous_parts(ch2_bestregions, ch2_regions, ch2_bestindices)

            segmodel = extract_segmentation_model(ch4_result)
            ch4_binary, sm = segment_sequence(ch4_result, segmodel = segmodel)
            ch4_regions, dum1, dum2 = extract_binary_regions(ch4_binary)
            #regions = split_up_binary_regions(regions, opening_param = 1, mshape = disk(3))#opening_param = 5)
            ch4_bestregions, ch4_bestindices = best_component_ch(ch4_regions)
            #ch4_bestregions = cut_superfluous_parts(ch4_bestregions, ch4_regions, ch4_bestindices)

            volume = np.pi/4./1000. * np.sum(np.sum(ch2_bestregions, axis=-1) * np.sum(ch4_bestregions, axis=-1), axis=-1)
            systole_est, diastole_est = np.min(volume,axis=0), np.max(volume,axis=0) # in square pixels
            # how much is one pixel^3?
            correction_factor = np.sqrt(np.abs(np.linalg.det(trf_4ch.params[:2,:2]))) * np.abs(np.linalg.det(trf_2ch.params[:2,:2])) * ch2_metadata["PixelSpacing"][0]**3

            systole_est = systole_est * correction_factor
            diastole_est = diastole_est * correction_factor

            print "******************************"
            print "patient:", patient_id
            print systole_est, diastole_est
            if patient_id-1<len(labels):
                print labels[patient_id-1,1], labels[patient_id-1,2]
                data_dump.append([systole_est, diastole_est, labels[patient_id-1,1], labels[patient_id-1,2]])
            else:
                data_dump.append([systole_est, diastole_est, None, None])
            print "******************************"
            pickle.dump(data_dump, open("segmentation.pkl", "wb"))

            if do_plot:
                fig = plt.figure()
                plt.subplot(221)
                def init_out1():
                    im1.set_data(ch2_bestregions[0])

                def animate_out1(i):
                    im1.set_data(ch2_bestregions[i])
                    return im1

                im1 = fig.gca().imshow(ch2_bestregions[0])
                fig.gca().set_aspect('equal')
                anim1 = animation.FuncAnimation(fig, animate_out1, init_func=init_out1, frames=30, interval=50)

                plt.subplot(222)

                def init_out2():
                    im2.set_data(ch4_bestregions[0])

                def animate_out2(i):
                    im2.set_data(ch4_bestregions[i])
                    return im2

                im2 = fig.gca().imshow(ch4_bestregions[0])
                fig.gca().set_aspect('equal')
                anim2 = animation.FuncAnimation(fig, animate_out2, init_func=init_out2, frames=30, interval=50)

                plt.subplot(223)

                def init_out3():
                    im3.set_data(ch2_result[0])

                def animate_out3(i):
                    im3.set_data(ch2_result[i])
                    return im3
                ch2_result[:,:,50] = 0
                im3 = fig.gca().imshow(ch2_result[0])
                fig.gca().set_aspect('equal')
                anim3 = animation.FuncAnimation(fig, animate_out3, init_func=init_out3, frames=30, interval=50)

                plt.subplot(224)

                def init_out4():
                    im4.set_data(ch4_result[0])

                def animate_out4(i):
                    im4.set_data(ch4_result[i])
                    return im4

                ch4_result[:,:,50] = 0
                im4 = fig.gca().imshow(ch4_result[0])
                fig.gca().set_aspect('equal')
                anim4 = animation.FuncAnimation(fig, animate_out4, init_func=init_out4, frames=30, interval=50)

                plt.show()
    else:
        data_dump = pickle.load(open("segmentation.pkl", "rb"))

    data_dump = np.array(data_dump)

    predictions = [{"patient": i+1,
                    "systole": np.zeros((0,600)),
                    "diastole": np.zeros((0,600))
                    } for i in xrange(700)]

    data_dump[data_dump > 600] = 600
    data_dump[data_dump < 0] = 0

    params = np.polyfit(data_dump[:,0], data_dump[:,2], 1)
    data_dump[:,0] = data_dump[:,0]*params[0]+params[1]
    error = np.abs(data_dump[:,0] - data_dump[:,2])
    params_error = np.polyfit(data_dump[:,0], error, 1)

    for i,v in enumerate(data_dump[:,0]):
        predictions[i]["systole"] = numpy_mu_sigma_erf(v, np.sqrt(params_error[0]*v+params_error[1]))[None,:]

    params = np.polyfit(data_dump[:,1], data_dump[:,3], 1)
    data_dump[:,1] = data_dump[:,1]*params[0]+params[1]
    error = np.abs(data_dump[:,1] - data_dump[:,3])
    params_error = np.polyfit(data_dump[:,1], error, 1)

    for i,v in enumerate(data_dump[:,1]):
        predictions[i]["diastole"] = numpy_mu_sigma_erf(v, np.sqrt(params_error[0]*v+params_error[1]))[None,:]


    # calculate CRPS
    crps = []
    for pred, label in zip(predictions[:500],labels):
        assert pred["patient"]==label[0]
        target_sys = np.zeros( (600,) , dtype='float32')
        target_sys[int(np.ceil(label[1])):] = 1  # don't forget to ceil!
        crps.append(np.mean( (pred["systole"][0] - target_sys)**2 ))
        target_dia = np.zeros( (600,) , dtype='float32')
        target_dia[int(np.ceil(label[2])):] = 1  # don't forget to ceil!
        crps.append(np.mean( (pred["diastole"][0] - target_dia)**2 ))

    print "crps:", np.mean(crps)

