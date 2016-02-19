#import numpy as np
#import skimage
import cPickle as pickle
import pylab as plt
import os
import numpy as np
import Joni_get_data as gd
import matplotlib
# matplotlib.use('Qt4Agg')
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion
from scipy.fftpack import fftn, ifftn
from scipy.signal import argrelmin, correlate
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import Joni_get_data as gd
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.draw import ellipse
from skimage.util import img_as_ubyte
from extract_roi import axis_likelyhood_surface
from extract_roi import sort_images
from extract_roi import align_images

# train_data_path='C:\\Users\\jdambre\\Werk\\Competitions and ML playground\\kaggle-heart-master\\segmentation\\segmentation\\data\\pkl_train\\'
# validate_data_path='C:\\Users\\jdambre\\Werk\\Competitions and ML playground\\kaggle-heart-master\\segmentation\\segmentation\\data\\pkl_train\\'

train_data_path='/home/lio/geit/data/dsb15_pkl/pkl_train/'
validate_data_path='/home/lio/geit/data/dsb15_pkl/pkl_valid/'


# problematic: /
data_index='8'

data=gd.get_data(train_data_path + data_index)
print len(data)
processdata=sort_images(data)
#processdata=gd.transform_data(processdata)  # do Jeroen-transforms
#processdata=align_images(processdata)

kernel_width = 5
center_margin = 8
num_peaks = 10
num_circles = 20
upscale = 1.5
minradius = 10
maxradius = 40
radstep = 2

lsurface, ROImask, ROIcenter, ROIradii = axis_likelyhood_surface(processdata, kernel_width = kernel_width, center_margin = center_margin, num_peaks = num_peaks, num_circles = num_circles, upscale = upscale, minradius = minradius, maxradius = maxradius, radstep = radstep)

x_axis=ROIcenter[0]
y_axis = ROIcenter[1]

print ROImask.shape, ROIcenter, processdata[0]['data'].shape


numslices = len(processdata)

for ddi in processdata:
    #indata=data[7]['data']
    outdata=ddi['data']

    #img_dc=np.mean(outdata,0)
    mdata=np.round(5.0*outdata)/5.0

    ff1 = fftn(outdata)
    first_harmonic1 = np.absolute(ifftn(ff1[1, :, :]))
    first_harmonic1[first_harmonic1 < 0.1* np.max(first_harmonic1)] =0.0
    second_harmonic1 = np.absolute(ifftn(ff1[2, :, :]))
    second_harmonic1[second_harmonic1 < 0.1* np.max(second_harmonic1)] =0.0


    image=img_as_ubyte(first_harmonic1/np.max(first_harmonic1))
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    # Detect two radii
    hough_radii = np.arange(minradius, maxradius, radstep)
    hough_res = hough_circle(edges, hough_radii)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    intensity=accums#(accums - mini)/(maxi-mini)

    # Draw the most prominent 5 circles
    ximagesize=processdata[0]['data'].shape[1]
    yimagesize=processdata[0]['data'].shape[2]
    image = color.gray2rgb(image)
    sorted=np.argsort(accums)[::-1][:num_circles]
    accs = []
    for idx in sorted:
        accs.append(accums[idx])
        center_x, center_y = centers[idx]
        radius = radii[idx]
        brightness=intensity[idx]
        cx, cy = circle_perimeter(center_y, center_x, radius)
        #dum=np.logical_and(cx < 128, cx < 128)
        dum=(cx < ximagesize) & (cy < yimagesize)
        cx=cx[dum]
        cy=cy[dum]
        #print cx.size, cy.size, dum.size, (np.round(brightness*255), 0, 0)
        image[cy, cx] = (np.round(brightness*255), 0, 0)
    #print accs

    # visualise everything
    for s in range(outdata.shape[0]):
            outdata[s][ROImask > 0.5] = 0.4 * outdata[s][ROImask > 0.5]

    for h in range(-4, 5, 1):
        image[x_axis,y_axis+h] = (0, 255, 0)
        outdata[:,x_axis,y_axis+h] = 0

    for v in range(-4, 5, 1):
        image[x_axis+v,y_axis] = (0, 255, 0)
        outdata[:,x_axis+v,y_axis] = 0


    fig = plt.figure(1)
    plt.subplot(221)
    fig.canvas.set_window_title('Original')
    def init_out():
        im2.set_data(outdata[0])

    def animate_out(i):
        im2.set_data(outdata[i])
        return im2

    im2 = fig.gca().imshow(outdata[0], cmap='gist_gray_r', vmin=0, vmax=255)
    anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)

    plt.subplot(223)
    fig.gca().imshow(first_harmonic1)#,cmap='gist_gray_r', vmin=0., vmax=1.)

    plt.subplot(222)
    fig.gca().imshow(image)#,cmap='gist_gray_r', vmin=0., vmax=1.)
    #ax.imshow(image, cmap=plt.cm.gray)

    plt.subplot(224)
    fig.gca().imshow(lsurface)#,cmap='gist_gray_r', vmin=0., vmax=1.)

    plt.show()

#dd=pickle.load(open('data/pkl_train/10/study/sax_12.pkl','rb'))
#data=dd["data"]
#metadata=dd["metadata"]
#for ind in range(30):
#    plt.subplot(4,8,ind+1)
#    plt.imshow(data[ind,:,:])
#plt.show()
#
#plt.subplot(1,2,1)
#meanImg=np.mean(data,0)
#plt.imshow(meanImg)
#plt.subplot(1,2,2)
#plt.imshow(data[29,:,:]-meanImg)
#
#plt.show()
#
#os.system("pause")


