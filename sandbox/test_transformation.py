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
patient_path = sorted(glob.glob(data_path + '/356/study'))
for p in patient_path:
    print p
    spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    for s in spaths:
        print s
        d = data_test.read_slice(s)
        print d.shape
        metadata = data_test.read_metadata(s)
        normalised_shape = tuple(int(float(d) * ps) for d, ps in zip(d.shape[1:], metadata['PixelSpacing']))
        print 'shape in mm', normalised_shape
        print '-----------------------------------'


        def init():
            im.set_data(d[0])


        def animate(i):
            im.set_data(d[i])
            return im


        fig = plt.figure(1)
        fig.canvas.set_window_title(s)
        plt.subplot(121)
        im = plt.gca().imshow(d[0], cmap='gist_gray_r', vmin=0, vmax=255)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30, interval=50)

        # ---------------------------------

        out_data = data_test.transform_norm_rescale(d, metadata, valid_transformation_params)


        def init_out():
            im2.set_data(out_data[0])


        def animate_out(i):
            im2.set_data(out_data[i])
            return im2


        plt.subplot(122)
        im2 = fig.gca().imshow(out_data[0], cmap='gist_gray_r', vmin=0., vmax=1.)
        anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)

        plt.show()
