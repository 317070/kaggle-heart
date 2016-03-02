import matplotlib.pyplot as plt
from matplotlib import animation

import load_data_batch
import cPickle as pickle
import os, glob, re
from extract_roi import *
import cv2


train_data_path='./pkl_train/'
validate_data_path='./pkl_validate'

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

##
# TODO:
#       involve whole sequence in region selection to mediate disagreements (morphological reconstruction?)
#       find other ways to disconnect connected regions: improve histogram or threshold selection?
#       clean up and extract remaining hyperparams
#       apply to 2ch and 4ch
##




# problematic: /
#data_index=1
#data_index=508
do_jeroen = False

data_index=1

data=load_data_batch.get_data(train_data_path,data_index)
#data=load_data_batch.get_data(validate_data_path,data_index)


processdata=sort_images(data) 


kernel_width = 5
center_margin = 8
num_peaks = 10
num_circles = 10  # 20
upscale = 1.5
minradius = 15
maxradius = 65
radstep = 2

patchsize = 32

lsurface, ROImask, ROIcenter, ROIaxes = axis_likelyhood_surface(processdata, kernel_width = kernel_width, center_margin = center_margin, num_peaks = num_peaks, num_circles = num_circles, upscale = upscale, minradius = minradius, maxradius = maxradius, radstep = radstep)
rs_minradius = max(int(minradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
rs_maxradius = max(int(maxradius / np.max(data[0]['metadata']['PixelSpacing'])),1)
x_center = ROIcenter[0]
y_center = ROIcenter[1]
x_axis = ROIaxes[0]
y_axis = ROIaxes[1]

numslices = len(processdata)
centerslice=numslices/2



slice_xdim, slice_ydim = processdata[centerslice]['data'].shape[1:]
slice_zdim = processdata[centerslice]['data'].shape[0]

testblock = processdata[centerslice]['data'].copy()
if not do_jeroen:
    testblock = testblock/1.0

# fit gmm
g, bloodlabel, lowerb, upperb = extract_segmentation_model(testblock)

# apply to slices - start region detection

for current_slice in range(numslices):





    slice_xdim, slice_ydim = processdata[current_slice]['data'].shape[1:]
    slice_zdim = processdata[current_slice]['data'].shape[0]

    testblock = processdata[current_slice]['data'].copy()

    if not do_jeroen:
        testblock = testblock/1.0
    labelblock = np.zeros_like(testblock)
    preprocessblock = np.zeros_like(testblock)



    for idx in range(slice_zdim):

        patch = testblock[idx]
       

        perc = 1.0*np.percentile(patch[patch>0], q=[10, 95])
        lowerb, upperb = perc[0], perc[1]
        rescalefact = 1.0
        patch[patch>0.] = (patch[patch>0.]-lowerb)/(upperb - lowerb)
        patch = patch * ROImask

        patch[patch<0] = 0.
        patch[patch>1] = 1.
        preprocessblock[idx] = patch.copy()
 
        

        compute optical flow for each slice 
        if idx > 0:
            patch=preprocessblock[idx]
            flow = cv2.calcOpticalFlowFarneback(previous_patch,patch,0.5,1, 3, 15, 3, 5, 1)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                
            hsv = np.zeros((patch.shape[0],patch.shape[1],3),np.uint8)
            hsv[...,1] = 255
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            rgb=np.zeros_like(bgr)
            rgb[...,0]=bgr[...,2]
            rgb[...,2]=bgr[...,0]
            rgb[...,1]=bgr[...,1]

            fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
            ax0, ax1, ax2 = axes

            ax0.imshow(previous_patch)
            ax0.set_title('previous_patch')
            ax1.imshow(patch)
            ax1.set_title('current_patch')
            ax2.imshow(rgb)
            ax2.set_title('optical flow')
            plt.show()

        thresh = np.zeros_like(patch)
        dum = patch.flatten()
        dum = np.reshape(dum,(dum.shape[0],1))
        thresh =g.predict(dum)
        thresh = np.reshape(thresh.T,patch.shape)
        binary = np.zeros_like(patch)
        binary[thresh == bloodlabel[1]] = 1

        xsurface=np.tile(range(slice_xdim),(slice_ydim,1)).T
        ysurface=np.tile(range(slice_ydim),(slice_xdim,1))
        distmat=np.sqrt(((xsurface-x_center)**2+(ysurface-y_center)**2))  # distance

        opening_param = 3
        mshape = square(3)
        binary = ndi.binary_erosion(binary,structure = mshape,iterations = opening_param)

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
   #     print "type of local maxi: " + str(type(local_maxi))
    #    print "local maxi: " + str(type(local_maxi))
       # plt.imshow(local_maxi)
      #  plt.show()
        local_maxi=ndi.binary_closing(local_maxi,structure = disk(4),iterations = 4)
      #  plt.imshow(local_maxi)
       # plt.show()
        #local_maxi = binary[1:-1,1:-1].copy()
        #vert = np.maximum(distance[:-2,1:-1],distance[2:,1:-1])
        #hor = np.maximum(distance[1:-1,:-2],distance[1:-1,2:])
        #dcenter = distance[1:-1,1:-1]

        #nomax = np.maximum(vert,hor) > dcenter
        #local_maxi[nomax] = 0

        markers = ndi.label(local_maxi)[0]

        #fig = plt.figure()
        #ax = Axes3D(fig)
        #ax.scatter(xsurface, ysurface, skeldist)
        #plt.show()



        #plt.figure()
        #plt.subplot(121)
        #plt.imshow(local_maxi)
        #plt.subplot(122)
        #plt.imshow(markers)
        #plt.show()

        labels = watershed(-distance, markers, mask=component3)
        component3, labeled_regions3 = select_component_sqdist(labels, np.max(labels), (x_center, y_center), distmat, opening_param = opening_param, do_diler = False)
        component3=ndi.binary_closing(component3,structure = disk(3),iterations = 2)

        labelblock[idx] = component3.copy()

        #fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        #ax0, ax1, ax2 = axes

        #ax0.imshow(component3, cmap=plt.cm.gray, interpolation='nearest')
        #ax0.set_title('Overlapping objects')
        #ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
        #ax1.set_title('Distances')
        #ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
        #ax2.set_title('Separated objects')

        #for ax in axes:
        #    ax.axis('off')

        #fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.9, bottom=0, left=0,right=1)

        #plt.show()

        #plt.figure()
        #plt.subplot(221)
        #plt.imshow(add_cross_to_image(patch, x_center, y_center))
        #plt.subplot(222)
        #plt.imshow(add_cross_to_image(component3, x_center, y_center))
        #plt.subplot(223)
        #plt.imshow(skeldist)
        #plt.subplot(224)
        #plt.imshow(labels)
        #plt.show()

        previous_patch=preprocessblock[idx]

    print str(current_slice)
    fig = plt.figure(1)
    plt.subplot(131)

    def init_out1():
        im1.set_data(testblock[0])

    def animate_out1(i):
        im1.set_data(testblock[i])
        return im1

    im1 = fig.gca().imshow(testblock[0])
    anim1 = animation.FuncAnimation(fig, animate_out1, init_func=init_out1, frames=30, interval=50)

    plt.subplot(132)

    def init_out2():
        im2.set_data(preprocessblock[0])

    def animate_out2(i):
        im2.set_data(preprocessblock[i])
        return im2

    im2 = fig.gca().imshow(preprocessblock[0])
    anim2 = animation.FuncAnimation(fig, animate_out2, init_func=init_out2, frames=30, interval=50)

    plt.subplot(133)

    def init_out3():
        im3.set_data(labelblock[0])

    def animate_out3(i):
        im3.set_data(labelblock[i])
        return im3

    im3 = fig.gca().imshow(labelblock[0])
    anim3 = animation.FuncAnimation(fig, animate_out3, init_func=init_out3, frames=30, interval=50)


    plt.show()




