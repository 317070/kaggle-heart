import matplotlib

matplotlib.use('Qt4Agg')

import glob
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import utils

print utils.get_dir_path('train')
filenames = glob.glob(utils.get_dir_path('train') + '/*rescale*.pkl')

for f in filenames:
    metadata = pickle.load(open(f))
    train_losses = metadata['losses_train']
    valid_losses = metadata['losses_eval_valid']
    valid_crpss = metadata['crps_eval_valid']

    for i in xrange(len(valid_losses)):
        print 'train loss', train_losses[i]
        print 'valid loss', valid_losses[i]
        print 'valid crpss', valid_crpss[i]
        print '-----------------'
    print 'chunks since start', metadata['chunks_since_start']

    for i in xrange(len(valid_losses)):
        valid_losses[i] = np.mean(valid_losses[i])

    fig = plt.figure()
    plt.title(f)
    x_range = np.arange(len(train_losses)) + 1

    plt.gca().set_yscale('log')
    plt.plot(x_range, train_losses)
    plt.plot(x_range, valid_losses)

    plt.ylabel("error")
    plt.show()
