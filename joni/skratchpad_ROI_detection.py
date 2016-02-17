import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn
from skimage import color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import glob
import re
import data_test
import copy

patch_size = (128, 128)
train_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': (-16, 16),
    'translation_range': (-8, 8),
    'shear_range': (0, 0),
    'do_flip': True,
    'sequence_shift': True
}

valid_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': None,
    'translation_range': None,
    'shear_range': None,
    'do_flip': None,
    'sequence_shift': None
}

kernel_width = 5
center_margin = 8
num_peaks = 10
num_circles = 20
upscale = 1.5
radstep = 2

# train_data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate/'
# validate_data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
data_path = '/mnt/sda3/CODING/python/kaggle-heart/data/train'


def get_data(pid_path):
    patient_path = sorted(glob.glob(pid_path + '/study'))
    in_data = list()
    for p in patient_path:
        spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
        print spaths
        print len(spaths)
        for s in spaths:
            metadata = data_test.read_metadata(s)
            data = data_test.read_slice(s)
            in_data.append({'data': data, 'metadata': metadata})
    return in_data


def transform_data(in_data):
    out_data = copy.deepcopy(in_data)
    for dd in out_data:
        dd['data'] = data_test.transform_norm_rescale(dd['data'], dd['metadata'], valid_transformation_params) * 255
    return out_data


def sort_slices(data):
    numslices = len(data)
    positions = np.zeros((numslices,))
    for idx in range(numslices):
        positions[idx] = data[idx]['metadata']['SliceLocation']
    newdata = [x for y, x in sorted(zip(positions.tolist(), data), key=lambda dd: dd[0], reverse=True)]
    return newdata


def plot(pid):
    original_data = get_data(data_path + '/' + str(pid))
    sorted_slices = sort_slices(original_data)

    maxradius = int(45 / sorted_slices[0]['metadata']['PixelSpacing'][0])
    minradius = int(10 / sorted_slices[0]['metadata']['PixelSpacing'][0])

    lsurface, roi_mask, roi_center = data_test.extract_roi_joni(sorted_slices, kernel_width=kernel_width,
                                                                center_margin=center_margin, num_peaks=num_peaks,
                                                                num_circles=num_circles, upscale=upscale,
                                                                minradius=minradius,
                                                                maxradius=maxradius, radstep=radstep)
    x_roicenter, y_roicenter = roi_center[0], roi_center[1]
    print pid
    print x_roicenter, y_roicenter

    print len(sorted_slices)
    for dslice in [sorted_slices[0], sorted_slices[len(sorted_slices) / 2], sorted_slices[-1]]:
        outdata = dslice['data']
        # print dslice['metadata']['SliceLocation']
        # print dslice['metadata']['ImageOrientationPatient']
        # print dslice['metadata']['PixelSpacing']
        # print dslice['data'].shape
        # print '--------------------------------------'

        ff1 = fftn(outdata)
        first_harmonic1 = np.absolute(ifftn(ff1[1, :, :]))
        first_harmonic1[first_harmonic1 < 0.1 * np.max(first_harmonic1)] = 0.0
        second_harmonic1 = np.absolute(ifftn(ff1[2, :, :]))
        second_harmonic1[second_harmonic1 < 0.1 * np.max(second_harmonic1)] = 0.0

        image = img_as_ubyte(first_harmonic1 / np.max(first_harmonic1))
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

        intensity = accums

        # Draw the most prominent 5 circles
        ximagesize = dslice['data'].shape[1]
        yimagesize = dslice['data'].shape[2]
        image = color.gray2rgb(image)
        sorted_circles = np.argsort(accums)[::-1][:num_circles]
        accs = []
        for idx in sorted_circles:
            accs.append(accums[idx])
            center_x, center_y = centers[idx]
            radius = radii[idx]
            brightness = intensity[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            dum = (cx < ximagesize) & (cy < yimagesize)
            cx = cx[dum]
            cy = cy[dum]
            image[cy, cx] = (np.round(brightness * 255), 0, 0)

        # visualise everything

        for s in range(outdata.shape[0]):
            outdata[s][roi_mask > 0.5] = 0.4 * outdata[s][roi_mask > 0.5]

        for h in range(-4, 5, 1):
            image[x_roicenter, y_roicenter + h] = (0, 255, 0)
            outdata[:, x_roicenter, y_roicenter + h] = 0

        for v in range(-4, 5, 1):
            image[x_roicenter + v, y_roicenter] = (0, 255, 0)
            outdata[:, x_roicenter + v, y_roicenter] = 0

        fig = plt.figure(1)
        plt.subplot(221)
        fig.canvas.set_window_title(pid)

        def init_out():
            im2.set_data(outdata[0])

        def animate_out(i):
            im2.set_data(outdata[i])
            return im2

        im2 = fig.gca().imshow(outdata[0], cmap='gist_gray_r', vmin=0, vmax=255)
        anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)

        plt.subplot(223)
        fig.gca().imshow(first_harmonic1)

        plt.subplot(222)
        fig.gca().imshow(image)

        plt.subplot(224)
        fig.gca().imshow(lsurface)

        plt.show()


for pid in [501, 502]:
    plot(pid)
