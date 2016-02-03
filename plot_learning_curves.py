import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import cPickle as pickle

filenames = glob.glob('/mnt/storage/metadata/kaggle-heart/train/ira/vgg_rms_sd_norm*.pkl')

for f in filenames:
    metadata = pickle.load(open(f))
    train_losses = metadata['losses_train']

    valid_losses = metadata['losses_eval_valid']
    valid_losses_sys = [v[0] for v in valid_losses]
    valid_losses_dst = [v[1] for v in valid_losses]

    valid_crpss = metadata['crps_eval_valid']
    valid_crpss_sys = [v[0] for v in valid_crpss]
    valid_crpss_dst = [v[1] for v in valid_crpss]

    print valid_losses_sys
    print valid_losses_dst
    print valid_crpss_sys
    print valid_crpss_dst

    print 'Expid', metadata['configuration']
    print 'chunks since start', metadata['chunks_since_start']
    print 'valid loss', valid_losses[-1]
    print 'train loss', train_losses[-1]
    print '===================================================='

    fig = plt.figure()
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(50, 100, 640, 545)
    plt.title(f)
    x_train = np.arange(len(train_losses)) + 1
    plt.gca().set_yscale('log')
    plt.plot(x_train, train_losses)
    x_valid = np.arange(0, len(valid_losses)) + 1
    plt.plot(x_valid, valid_losses_sys)
    plt.plot(x_valid, valid_losses_dst)
    plt.show()
