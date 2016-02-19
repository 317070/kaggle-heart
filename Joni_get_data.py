import matplotlib
# matplotlib.use('Qt4Agg')

import glob
import re
from matplotlib import animation
import matplotlib.pyplot as plt
import data as data_test
import os
import copy
from skimage import io, transform
import numpy as np

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



# data_path='C:\\Users\\jdambre\\Werk\\Competitions and ML playground\\kaggle-heart-master\\segmentation\\segmentation\\data\\pkl_train\\1'
def get_data(pp):
    patient_path = sorted(glob.glob(pp + '/study'))
    in_data=list()
    for p in patient_path:
        #print p
        spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'\w*_(\d+)*\.pkl$', x).group(1)))
        #for s in [spaths[0]]:
        for s in spaths:
            #print s
            data = data_test.read_slice(s)
            metadata = data_test.read_metadata(s)
            in_data.append({'data': data, 'metadata': metadata})
    return in_data

def transform_data(in_data):
    out_data=copy.deepcopy(in_data)
    for dd in out_data:
        dd['data']= data_test.transform_with_jeroen(dd['data'], dd['metadata'], valid_transformation_params)

    return out_data

def animate_data(in_data,out_data):
    data = in_data['data']
    metadata = in_data['data']

    def init():
        im.set_data(data[0])


    def animate(i):
        im.set_data(data[i])
        return im


    fig = plt.figure()
    fig.canvas.set_window_title('test')
    plt.subplot(121)
    im = plt.gca().imshow(data[0], cmap='gist_gray_r', vmin=0, vmax=255)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30, interval=50)

    # ---------------------------------

    data2= out_data['data']

    def init_out():
        im2.set_data(data2[0])


    def animate_out(i):
        im2.set_data(data2[i])
        return im2


    plt.subplot(122)
    im2 = fig.gca().imshow(data2[0], cmap='gist_gray_r', vmin=0., vmax=1.)
    anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)

    plt.show()
    os.system("pause")

def alignment_transform(sax1,sax2):

    source = np.mean(sax1['data'],0)
    destination = np.mean(sax2['data'],0)
    T = transform.estimate_transform('similarity',source,destination)
    return T