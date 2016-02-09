import matplotlib
import glob
import re
import cPickle as pickle
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import data_test

pid = 519
slice_id = 0
data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
patch_size = (128, 128)
train_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': (-16, 16),
    'translation_range': (-8, 8),
    'shear_range': (0, 0)
}

valid_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': None,
    'translation_range': None,
    'shear_range': None
}

patient_path = data_path + '/%s/study/' % pid
spaths = sorted(glob.glob(patient_path + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))

data = data_test.read_slice(spaths[slice_id])
metadata = data_test.read_metadata(spaths[slice_id])

print data.shape
print metadata


def init():
    im.set_data(data[0])


def animate(i):
    im.set_data(data[i])
    return im


fig = plt.figure()
im = fig.gca().imshow(data[0], cmap='gist_gray_r', vmin=0, vmax=255)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30, interval=50)

# ---------------------------------


out_data = data_test.fix_image_orientation(data, metadata)
print out_data.shape

# dd = data_test.normalize_contrast(np.copy(data))
# out_data, pixelspacing = data_test.transform_with_metadata(dd, metadata, valid_transformation_params)
# print out_data.shape
# print pixelspacing


def init_out():
    im2.set_data(out_data[0])


def animate_out(i):
    im2.set_data(out_data[i])
    return im2


fig = plt.figure()
im2 = fig.gca().imshow(out_data[0], cmap='gist_gray_r', vmin=0, vmax=255)
anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)

plt.show()
