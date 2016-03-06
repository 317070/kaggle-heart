# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import glob
import os
import data
import data_prep_ch
import re
import utils
import matplotlib
# matplotlib.use('Qt4Agg')

import data
import numpy as np
import glob
import re
from matplotlib import animation
import matplotlib.pyplot as plt
import utils
import data as data_test
from configuration import set_configuration, config

set_configuration('test_config')
patch_size = config().patch_size
train_transformation_params = config().train_transformation_params
valid_transformation_params = config().valid_transformation_params

data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
slice2roi = utils.load_pkl('../pkl_train_slice2roi.pkl')
slice2roi_valid = utils.load_pkl('../pkl_validate_slice2roi.pkl')
slice2roi.update(slice2roi_valid)

patient_path = sorted(glob.glob(data_path + '/*/study'))
for p in patient_path:
    print p
    sax_spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    slicepath2metadata = {}
    slicepath2roi = {}

    for s in sax_spaths:
        metadata = data_test.read_metadata(s)
        slicepath2metadata[s] = metadata
        pid = utils.get_patient_id(s)
        sid = utils.get_slice_id(s)
        roi = slice2roi[pid][sid]
        slicepath2roi[s] = roi

    ch2_path = glob.glob(p + '/2ch_*.pkl')
    if ch2_path:
        data_ch2 = data.read_slice(ch2_path[0])
        metadata_ch2 = data.read_metadata(ch2_path[0])
    else:
        data_ch2, metadata_ch2 = None, None

    ch4_path = glob.glob(p + '/4ch_*.pkl')
    if ch4_path:
        data_ch4 = data.read_slice(ch4_path[0])
        metadata_ch4 = data.read_metadata(ch4_path[0])
    else:
        data_ch4, metadata_ch4 = None, None

    in_data = data_ch4

    def init():
        im.set_data(in_data[0])


    def animate(i):
        im.set_data(in_data[i])
        return im


    fig = plt.figure(1)
    fig.canvas.set_window_title(p)
    plt.subplot(121)
    im = plt.gca().imshow(in_data[0], cmap='gist_gray_r')
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(in_data), interval=50)

    # ---------------------------------
    out_data_ch2, out_data_ch4, _ = data_test.transform_ch(data_ch2, metadata_ch2, data_ch4, metadata_ch4,
                                                           slicepath2metadata, train_transformation_params,
                                                           sax2roi=slicepath2roi)
    out_data = out_data_ch4


    def init_out():
        im2.set_data(out_data[0])


    def animate_out(i):
        im2.set_data(out_data[i])
        return im2


    plt.subplot(122)
    im2 = fig.gca().imshow(out_data[0], cmap='gist_gray_r')
    anim2 = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=len(out_data), interval=50)

    plt.show()
