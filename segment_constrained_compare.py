import matplotlib.pyplot as plt
from matplotlib import animation

#import Joni_get_data as gd

import load_data_batch
import cPickle as pickle
import os, glob, re
from extract_roi import *

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


data_index=55

# if data_index>500:
#     data=gd.get_data(validate_data_path + str(data_index))
#     ch2=gd.get_data(validate_data_path + str(data_index), tag = "2ch")
#     ch4=gd.get_data(validate_data_path + str(data_index), tag = "4ch")
# else:
#     data=gd.get_data(train_data_path + str(data_index))
#     ch2=gd.get_data(train_data_path + str(data_index), tag = "2ch")
#     ch4=gd.get_data(train_data_path + str(data_index), tag = "4ch")



#data=load_data_batch.get_data(train_data_path,data_index)
data=load_data_batch.get_data(validate_data_path,data_index)


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

# assemble histogram and fit GMM


g, bloodlabel, lowerb, upperb = extract_segmentation_model(testblock)

# apply to slices - start region detection

for current_slice in range(numslices):

    slice_xdim, slice_ydim = processdata[current_slice]['data'].shape[1:]
    slice_zdim = processdata[current_slice]['data'].shape[0]

    testblock = processdata[current_slice]['data'].copy()
    #labelblock = np.zeros_like(testblock)
    if not do_jeroen:
        testblock = testblock/1.0
    labelblock = np.zeros_like(testblock)
    labelblock_constrained = np.zeros_like(testblock)
    preprocessblock = np.zeros_like(testblock)

    for idx in range(slice_zdim):
        patch = testblock[idx]
        perc = 1.0*np.percentile(patch[patch>0], q=[10, 95])
        lowerb, upperb = perc[0], perc[1]
        rescalefact = 1.0
        patch[patch>0.] = (patch[patch>0.]-lowerb)/(upperb - lowerb)
        patch = patch * ROImask
        #patch[patch>0] = (patch[patch>0]-lowerb)/(upperb - lowerb)
        patch[patch<0] = 0.
        patch[patch>1] = 1.

        preprocessblock[idx] = patch.copy()
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

        #component2, labeled_regions2 = select_component_sqdist(binary, (x_center, y_center), distmat, opening_param = 3)
        opening_param = 3
        mshape = square(3)
        binary = ndi.binary_erosion(binary,structure = mshape,iterations = opening_param)

        
        # split up independent regions
        labeled_regions, num_labels = ndi.label(binary)
        labeled_regions_constrained=labeled_regions.copy(); 
        
 
    
        for label in range(1,num_labels+1):
            indices=np.nonzero(labeled_regions==label)

            region_mean_y=np.mean(indices[1])
            region_mean_x=np.mean(indices[0])
            region_min=np.min(indices[1])
        

            if(region_mean_y < (1.0-0.07)*y_center or region_mean_y > (1.0+0.07)*y_center or  region_mean_x < (1.0 - 0.1)*x_center or region_mean_x > (1.0 + 0.1)*x_center):
                labeled_regions_constrained[labeled_regions_constrained==label] = 0


    
        component3, labeled_regions3 = select_component_sqdist(labeled_regions, num_labels, (x_center, y_center), distmat, opening_param = opening_param, do_diler = True)
        component3_constrained, labeled_regions3_constrained = select_component_sqdist(labeled_regions_constrained, num_labels, (x_center, y_center), distmat, opening_param = opening_param, do_diler = True)
 
     
        labelblock[idx] = component3.copy()
        labelblock_constrained[idx] = component3_constrained.copy()
       
    

        mshape = square(3)
        #component3=ndi.binary_closing(component3,structure = ((1,0,0),(0,1,0),(0,0,1)),iterations = 3)
        distance = ndi.distance_transform_edt(component3)
        distance_constrained = ndi.distance_transform_edt(component3_constrained)
        skel = skeletonize(component3)
        skel_constrained = skeletonize(component3_constrained)
    
 
        skeldist = distance*skel
        skeldist_constrained = distance_constrained*skel_constrained

        md = rs_maxradius
        #local_maxi = filters.maximum_filter(skeldist, footprint=np.ones((3, 3)))
        local_maxi = peak_local_max(skeldist, indices=False, footprint=disk(10))
        local_maxi_constrained = peak_local_max(skeldist_constrained, indices=False, footprint=disk(10))
  
        local_maxi=ndi.binary_closing(local_maxi,structure = disk(4),iterations = 4)
        local_maxi_constrained=ndi.binary_closing(local_maxi_constrained,structure = disk(4),iterations = 4)
  
        #local_maxi = binary[1:-1,1:-1].copy()
        #vert = np.maximum(distance[:-2,1:-1],distance[2:,1:-1])
        #hor = np.maximum(distance[1:-1,:-2],distance[1:-1,2:])
        #dcenter = distance[1:-1,1:-1]

        #nomax = np.maximum(vert,hor) > dcenter
        #local_maxi[nomax] = 0

        markers = ndi.label(local_maxi)[0]
        markers_constrained = ndi.label(local_maxi_constrained)[0]



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
        labels_constrained = watershed(-distance_constrained, markers_constrained, mask=component3_constrained)


        component3, labeled_regions3 = select_component_sqdist(labels, np.max(labels), (x_center, y_center), distmat, opening_param = opening_param, do_diler = False)
        component3=ndi.binary_closing(component3,structure = disk(3),iterations = 2)

        component3_constrained, labeled_regions3_constrained = select_component_sqdist(labels_constrained, np.max(labels_constrained), (x_center, y_center), distmat, opening_param = opening_param, do_diler = False)
        component3_constrained=ndi.binary_closing(component3_constrained,structure = disk(3),iterations = 2)



        labelblock[idx] = component3.copy()
        labelblock_constrained[idx] = component3_constrained.copy()

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

  #  print str(current_slice)
    fig = plt.figure(1)
    plt.subplot(141)

    def init_out1():
        im1.set_data(testblock[0])

    def animate_out1(i):
        im1.set_data(testblock[i])
        return im1

    im1 = fig.gca().imshow(testblock[0])
    anim1 = animation.FuncAnimation(fig, animate_out1, init_func=init_out1, frames=30, interval=50)

    plt.subplot(142)

    def init_out2():
        im2.set_data(preprocessblock[0])

    def animate_out2(i):
        im2.set_data(preprocessblock[i])
        return im2

    im2 = fig.gca().imshow(preprocessblock[0])
    anim2 = animation.FuncAnimation(fig, animate_out2, init_func=init_out2, frames=30, interval=50)

    plt.subplot(143)

    def init_out3():
        im3.set_data(labelblock[0])

    def animate_out3(i):
        im3.set_data(labelblock[i])
        return im3

    im3 = fig.gca().imshow(labelblock[0])
    anim3 = animation.FuncAnimation(fig, animate_out3, init_func=init_out3, frames=30, interval=50)


    plt.subplot(144)

    def init_out4():
        im4.set_data(labelblock_constrained[0])

    def animate_out4(i):
        im4.set_data(labelblock_constrained[i])
        return im4

    im4 = fig.gca().imshow(labelblock_constrained[0])
    anim4 = animation.FuncAnimation(fig, animate_out4, init_func=init_out4, frames=30, interval=50)



    plt.show()




