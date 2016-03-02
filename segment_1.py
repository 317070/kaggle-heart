import matplotlib.pyplot as plt
from matplotlib import animation

import load_data_batch
import cPickle as pickle
import os, glob, re
from extract_roi import *

from scipy import ndimage as ndi
import scipy.ndimage.filters as filters

from skimage.morphology import watershed, skeletonize, square, disk
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.feature import peak_local_max
from math import sqrt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from skimage.morphology import square
from skimage.morphology import disk
from scipy import signal


def segment_slices(slice,x_center,y_center,x_axis,y_axis,mask,gmm_model,lowerb,upperb,bloodlabel,rs_maxradius):


   

    slice_xdim, slice_ydim =slice['data'].shape[1:]
    slice_zdim = slice['data'].shape[0]

    # assemble histogram and fit GMM



    #print "bloodlabel"+str(bloodlabel)

    # apply to slices - start region detection

    testblock = slice['data'].copy()#   
    testblock = testblock/1.0
    labelblock = np.zeros_like(testblock)
    preprocessblock = np.zeros_like(testblock)
 
    for idx in range(slice_zdim):
        patch = testblock[idx]
        perc = 1.0*np.percentile(patch[patch>0], q=[10, 95])
        lowerb, upperb = perc[0], perc[1]
        rescalefact = 1.0
        patch[patch>0.] = (patch[patch>0.]-lowerb)/(upperb - lowerb)
        #create roimask here because some images seem to have varying dimensions
        #can we output some patient id in case in fails again?
        #        Traceback (most recent call last):
        #   File "slice2roi.py", line 274, in <module>
        #     get_slice2roi(d, plot=False)
        #   File "slice2roi.py", line 203, in get_slice2roi
        #     mask=segment.segment_slices(slice, roi_center[0],roi_center[1],roi_radii[0],roi_radii[1],ROImask,g,lowerb,upperb,bloodlabel,rs_maxradius)
        #   File "/home/matthias/Documents/playground/heart-challenge/segmentation/segment.py", line 73, in segment_slices
        #     patch = patch * mask
        # ValueError: operands could not be broadcast together with shapes (192,256) (218,256) 
  
        ROImask = np.zeros_like(patch)
        ximagesize=ROImask.shape[0]
        yimagesize=ROImask.shape[1]
        [rr,cc] = ellipse(x_center, y_center, x_axis, y_axis)
        dum=(rr < ximagesize) & (rr>=0) & (cc < yimagesize) & (cc>=0)
        rr=rr[dum]
        cc=cc[dum]
        ROImask[rr,cc]=1.


        patch = patch * ROImask
        #patch[patch>0] = (patch[patch>0]-lowerb)/(upperb - lowerb)
        patch[patch<0] = 0.
        patch[patch>1] = 1.

        preprocessblock[idx] = patch.copy()
        thresh = np.zeros_like(patch)
        dum = patch.flatten()
        dum = np.reshape(dum,(dum.shape[0],1))
        thresh =gmm_model.predict(dum)
        thresh = np.reshape(thresh.T,patch.shape)
        binary = np.zeros_like(patch)
        binary[thresh == bloodlabel[1]] = 1

        xsurface=np.tile(range(slice_xdim),(slice_ydim,1)).T
        ysurface=np.tile(range(slice_ydim),(slice_xdim,1))
        distmat=np.sqrt(((xsurface-x_center)**2+(ysurface-y_center)**2))  # distance
 #       plt.imshow(binary)
  #      plt.show()

        #component2, labeled_regions2 = select_component_sqdist(binary, (x_center, y_center), distmat, opening_param = 3)
        opening_param = 3
        mshape = square(3)
        binary = ndi.binary_erosion(binary,structure = mshape,iterations = opening_param)

       # plt.imshow(binary)
       # plt.show()



        # split up independent regions
        labeled_regions, num_labels = ndi.label(binary)
        labeled_regions_1=labeled_regions.copy(); 

        for label in range(1,num_labels+1):
                indices=np.nonzero(labeled_regions==label)

                region_mean_x=np.mean(indices[1])
                region_mean_y=np.mean(indices[0])
                region_min=np.min(indices[1])

                if(region_mean_x < (1.0-0.07)*y_center or region_mean_x > (1.0+0.07)*y_center or  region_mean_y < (1.0 - 0.1)*x_center or region_mean_y > (1.0 + 0.1)*x_center):
                    labeled_regions_1[labeled_regions_1==label] = 0




#        component3, labeled_regions3 = select_component_sqdist(labeled_regions, num_labels, (x_center, y_center), distmat, opening_param = opening_param, do_diler = True)
        component3, labeled_regions3 = select_component_sqdist(labeled_regions_1, num_labels, (x_center, y_center), distmat, opening_param = opening_param, do_diler = True)
        labelblock[idx] = component3.copy()


        mshape = square(3)

        distance = ndi.distance_transform_edt(component3)
        skel = skeletonize(component3)

        skeldist = distance*skel
        md = rs_maxradius

        local_maxi = peak_local_max(skeldist, indices=False, footprint=disk(10))
        local_maxi=ndi.binary_closing(local_maxi,structure = disk(4),iterations = 4)

        markers = ndi.label(local_maxi)[0]


        labels = watershed(-distance, markers, mask=component3)
        component3, labeled_regions3 = select_component_sqdist(labels, np.max(labels), (x_center, y_center), distmat, opening_param = opening_param, do_diler = False)
        component3=ndi.binary_closing(component3,structure = disk(3),iterations = 2)

        labelblock[idx] = component3.copy()


    return labelblock

data_index=5



kernel_width = 5
center_margin = 8
num_peaks = 10
num_circles = 10  # 20
upscale = 1.5
minradius = 15
maxradius = 65
radstep = 2

patchsize = 32




if __name__ == '__main__':

    data_paths = ['./pkl_train', './pkl_validate']
    # data_paths = ['/mnt/sda3/data/kaggle-heart/pkl_validate']
    for data_path in data_paths:
        data=load_data_batch.get_data(data_path,data_index)
        processdata=sort_images(data) 
     #   get_slice2roi(d, plot=False)
        lsurface, ROImask, ROIcenter, ROIaxes = axis_likelyhood_surface(processdata, kernel_width = kernel_width, center_margin = center_margin, num_peaks = num_peaks, num_circles = num_circles, upscale = upscale, minradius = minradius, maxradius = maxradius, radstep = radstep)
        rs_minradius = max(int(minradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
        rs_maxradius = max(int(maxradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
        x_center = ROIcenter[0]
        y_center = ROIcenter[1]
        x_axis = ROIaxes[0]
        y_axis = ROIaxes[1]

                
        numslices = len(processdata)
        centerslice=numslices/2
        testblock = processdata[0]['data'].copy()#   

        g, bloodlabel, lowerb, upperb = extract_segmentation_model(testblock)
        for sliceidx in range(0,numslices):
            segment_slices(processdata[sliceidx],ROIcenter[0],ROIcenter[1],ROIaxes[0],ROIaxes[1],ROImask,g,lowerb,upperb,bloodlabel,rd_maxradius)
 

 
