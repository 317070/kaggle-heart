import matplotlib
# matplotlib.use('Qt4Agg')

import glob
import re
from matplotlib import animation
import matplotlib.pyplot as plt
import data_test

data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
# data_path = '/data/dsb15_pkl/pkl_train'
patient_path = sorted(glob.glob(data_path + '/*/study'))
for p in patient_path:
    spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    img_sizes = []
    if len(spaths) < 4:
        print p
        print len(spaths)
    # for s in spaths:
    #     data = data_test.read_slice(s)
    #     img_sizes.append(data.shape)
    #     metadata = data_test.read_metadata(s)
    #
    # img_ss = set(img_sizes)
    # if len(img_ss) != 1:
    #     print img_ss
    #     print p



