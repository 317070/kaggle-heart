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

data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
patient_path = glob.glob(data_path + '/*/study')
for p in patient_path:
    print p
    spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    for s in spaths:
        data = data_test.read_slice(s)
        metadata = data_test.read_metadata(s)


        def init():
            im.set_data(data[0])


        def animate(i):
            im.set_data(data[i])
            return im


        fig = plt.figure(1)
        fig.canvas.set_window_title(s)
        plt.subplot(121)
        im = plt.gca().imshow(data[0], cmap='gist_gray_r', vmin=0, vmax=255)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30, interval=50)

        # ---------------------------------

        out_data = data_test.fix_image_orientation(data, metadata)
        out_data = data_test.normalize_contrast(out_data)


        def init_out():
            im2.set_data(out_data[0])


        def animate_out(i):
            im2.set_data(out_data[i])
            return im2

        plt.subplot(122)
        im2 = fig.gca().imshow(out_data[0], cmap='gist_gray_r', vmin=0., vmax=1.)
        anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)

        plt.show()
