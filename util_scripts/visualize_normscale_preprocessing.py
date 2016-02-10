
import cPickle as pickle
import glob
import os

import dicom
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from pkl2patient import clean_metadata

import configuration
import data_loader

configuration.set_configuration('je_test')
_config = configuration.config


def extract_image_patch(chunk_dst, img):
    """
    extract a correctly sized patch from img and place it into chunk_dst,
    which assumed to be preinitialized to zeros.
    """
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 127
    # img[-1, :] = 127
    # img[:, 0] = 127
    # img[:, -1] = 127

    p_x, p_y = chunk_dst.shape
    im_x, im_y = img.shape

    offset_x = (im_x - p_x) // 2
    offset_y = (im_y - p_y) // 2

    if offset_x < 0:
        cx = slice(-offset_x, -offset_x + im_x)
        ix = slice(0, im_x)
    else:
        cx = slice(0, p_x)
        ix = slice(offset_x, offset_x + p_x)

    if offset_y < 0:
        cy = slice(-offset_y, -offset_y + im_y)
        iy = slice(0, im_y)
    else:
        cy = slice(0, p_y)
        iy = slice(offset_y, offset_y + p_y)

    chunk_dst[cx, cy] = img[ix, iy]


def extract_image_patch_left(chunk_dst, img):
    """
    extract a correctly sized patch from img and place it into chunk_dst,
    which assumed to be preinitialized to zeros.
    """
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 127
    # img[-1, :] = 127
    # img[:, 0] = 127
    # img[:, -1] = 127

    p_x, p_y = chunk_dst.shape
    im_x, im_y = img.shape

    offset_x = (im_x - p_x) // 2
    offset_y = 0

    if offset_x < 0:
        cx = slice(-offset_x, -offset_x + im_x)
        ix = slice(0, im_x)
    else:
        cx = slice(0, p_x)
        ix = slice(offset_x, offset_x + p_x)

    if offset_y < 0:
        cy = slice(-offset_y, -offset_y + im_y)
        iy = slice(0, im_y)
    else:
        cy = slice(0, p_y)
        iy = slice(offset_y, offset_y + p_y)

    print ix
    print iy
    chunk_dst[cx, cy] = img[ix, iy]


def animate_slice(slicedata1, slicedata2, index1, index2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(50, 100, 600, 300)

    im1 = ax1.imshow(slicedata1[0], cmap='gist_gray_r')
    im2 = ax2.imshow(slicedata2[0], cmap='gist_gray_r')

    fig.suptitle('patient %d vs %d' % (index1, index2))

    def init():
        im1.set_data(slicedata1[0])
        im2.set_data(slicedata2[0])

    def animate(i):
        im1.set_data(slicedata1[i])
        im2.set_data(slicedata2[i])
        return im1

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(slicedata1), interval=50)

    plt.show()


def animate_slice_crop(raw_slicedata, cropped_slicedata, index):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(50, 100, 600, 300)

    im1 = ax1.imshow(raw_slicedata[0], cmap='gist_gray_r')
    im2 = ax2.imshow(cropped_slicedata[0], cmap='gist_gray_r')

    fig.suptitle('patient %d' % (index,))

    def init():
        im1.set_data(raw_slicedata[0])
        im2.set_data(cropped_slicedata[0])

    def animate(i):
        im1.set_data(raw_slicedata[i])
        im2.set_data(cropped_slicedata[i])
        return im1

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(raw_slicedata), interval=50)

    plt.show()


# 328 vs 295
# largest normalised: 67 -> id 83
# smallest normalised: 54 -> id 66

crop_size = (30, 200, 200)

wanted_input_tags = ['sliced:data:singleslice:middle']
wanted_output_tags = ['systole', 'diastole', 'patients']
indices = [114]
print 'processing %d' % indices[0]
result_67 = data_loader.get_patient_data(
    indices, wanted_input_tags, wanted_output_tags, set="train",
    preprocess_function=_config().preprocess_train)
raw_slice_67 = result_67['input']['sliced:data:singleslice:middle:raw_0']
patient_id_67 = result_67['output']['patients'][0]

sizes = []
for i in range(0, 500):
#    print 'Loading and processing patient %d' % i
    indices = [i]
    result = data_loader.get_patient_data(
        indices, wanted_input_tags, wanted_output_tags, set="train",
        preprocess_function=_config().preprocess_train)
    raw_slice = result['input']['sliced:data:singleslice:middle:raw_0']
    patient_id = result['output']['patients'][0]

    raw_slice_patch = np.zeros(crop_size)
    for im_dst, im in zip(raw_slice_patch, raw_slice):
        extract_image_patch_left(im_dst, im)
    shape = raw_slice.shape[-2:]
    if shape[0] > shape[1]: shape = (shape[1], shape[0])
    print i, patient_id, shape
    sizes.append(raw_slice.shape[-2:])

    #animate_slice_crop(raw_slice, raw_slice_patch, patient_id)


# make scatterplot
x, y = zip(*sizes)
plt.scatter(x, y)
plt.show()
