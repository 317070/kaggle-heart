import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import cPickle as pickle

files = glob.glob('/mnt/storage/metadata/kaggle-heart/train/ira/*.pkl')

for f in files:
    metadata = pickle.load(open(f, "r"))
    train_losses = metadata['losses_train']
    valid_losses = metadata['losses_eval_valid']

    print 'Expid', metadata['configuration_file']
    print 'chunks since start', metadata['chunks_since_start']
    # print 'valid loss', valid_losses[-1]
    print 'train loss', train_losses[-1]
    print '===================================================='

    # fig = plt.figure()
    #
    # mngr = plt.get_current_fig_manager()
    # # to put it into the upper left corner for example:
    # mngr.window.setGeometry(50, 100, 640, 545)
    # plt.title(filename)
    # x_train = np.arange(len(train_losses)) + 1
    #
    # plt.gca().set_yscale('log')
    # plt.plot(x_train, train_losses)
    # if len(valid_losses) >= 1:
    #     x_valid = np.arange(0, len(train_losses), 1.0 * len(train_losses) / len(valid_losses)) + 1
    #     plt.plot(x_valid, valid_losses)
    # plt.show()

    print "done"
