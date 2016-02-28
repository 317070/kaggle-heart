import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.special import erf, erfinv

import cPickle as pickle
import glob
import os
import scipy
import scipy.ndimage.interpolation
#print glob.glob(os.path.expanduser("~/storage/metadata/kaggle-heart/predictions/j7_jeroen_ch.pkl"))
#predictions = pickle.load(open(glob.glob(os.path.expanduser("~/storage/metadata/kaggle-heart/predictions/j7_jeroen_ch.pkl"))[0]))["predictions"]
#scipy.ndimage.interpolation.zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0, prefilter=True)

p = np.array(range(0,600), dtype='float32')

predictions = (erf( (p - 300)/50 )+1)/2

def zoom(array, zoom_factor):
    result = np.ones(array.shape)
    zoom = [1.0]*array.ndim
    zoom[-1] = zoom_factor
    zr = scipy.ndimage.interpolation.zoom(array,
                                          zoom,
                                          order=3,
                                          mode='nearest',
                                          prefilter=True)
    result[...,:min(zr.shape[-1],array.shape[-1])] = zr[...,:min(zr.shape[-1],array.shape[-1])]
    return result


fig = plt.figure()
mngr = plt.get_current_fig_manager()
# to put it into the upper left corner for example:
mngr.window.setGeometry(50, 100, 600, 300)

im1 = fig.gca().plot(p, predictions)

def init():
    pp = predictions
    im1[0].set_ydata(pp)

def animate(i):
    z = float(i)/50
    pp = zoom(predictions,z)
    fig.suptitle("zoom %f"%z)
    im1[0].set_ydata(pp)
    return im1

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50)
#anim.save('my_animation.mp4')
plt.show()