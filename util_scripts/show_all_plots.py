import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import cPickle as pickle

print "Looking for the metadata files..."
files = sorted(glob.glob(os.path.expanduser("~/storage/metadata/kaggle-heart/train/*crps4.pkl")))
print "Plotting..."

fig = plt.figure()

mngr = plt.get_current_fig_manager()
# to put it into the upper left corner for example:
mngr.window.setGeometry(50, 100, 640, 545)
for file in files:
    try:
        filename = os.path.basename(os.path.normpath(file))
        data = pickle.load(open(file, "r"))
        train_losses = data['losses_train']
        valid_losses = data['losses_eval_valid']
        kaggle_losses = data['losses_eval_valid_kaggle']

        x_train = np.arange(len(train_losses))+1

        plt.gca().set_yscale('log')
        #plt.plot(x_train, train_losses)
        if len(valid_losses)>=1:
            x_valid = np.arange(0,len(train_losses),1.0*len(train_losses)/len(valid_losses))+1
            plt.plot(x_valid, valid_losses)
            #x_kaggle = np.arange(0,len(train_losses),1.0*len(train_losses)/len(kaggle_losses))+1
            #plt.plot(x_kaggle, kaggle_losses)

        plt.xlabel("chunks")
        plt.ylabel("error")
        print "min kaggle loss:", min(valid_losses)
        print "end kaggle loss:", valid_losses[-1]
    except:
        print "%s is corrupt. Skipping" % file

plt.show()

print "done"