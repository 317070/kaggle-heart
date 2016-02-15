import matplotlib
# matplotlib.use('Qt4Agg')

import glob
import re
from matplotlib import animation
import matplotlib.pyplot as plt
import data_test


patch_size = (128, 128)
train_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': (-16, 16),
    'translation_range': (-8, 8),
    'shear_range': (0, 0),
    'do_flip': True,
    'sequence_shift': False
}

valid_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': None,
    'translation_range': None,
    'shear_range': None,
    'do_flip': None,
    'sequence_shift': None
}

data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
# data_path = '/data/dsb15_pkl/pkl_train'
patient_path = sorted(glob.glob(data_path + '/*/study'))
# patient_path = [data_path + '/555/study', data_path+ '/693/study']
for p in patient_path:
    print p
    spaths = sorted(glob.glob(p + '/2ch_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    for s in spaths:
        print s
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

        out_data = data_test.transform_with_jeroen(data, metadata, valid_transformation_params)


        def init_out():
            im2.set_data(out_data[0])


        def animate_out(i):
            im2.set_data(out_data[i])
            return im2


        plt.subplot(122)
        im2 = fig.gca().imshow(out_data[0], cmap='gist_gray_r', vmin=0., vmax=1.)
        anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)

        plt.show()
