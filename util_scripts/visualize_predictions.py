import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import cPickle as pickle
import glob
import os
print glob.glob(os.path.expanduser("~/storage/metadata/kaggle-heart/predictions/j7_jeroen_ch.pkl"))
predictions = pickle.load(open(glob.glob(os.path.expanduser("~/storage/metadata/kaggle-heart/predictions/j7_jeroen_ch.pkl"))[0]))["predictions"]

print len(predictions)

p = np.linspace(0.0, 600.0, 600)
print predictions[0]["systole"].shape
pp = predictions[0]["systole"][0]

fig = plt.figure()
mngr = plt.get_current_fig_manager()
# to put it into the upper left corner for example:
mngr.window.setGeometry(50, 100, 600, 300)

im1 = fig.gca().plot(p, pp)

def init():
    pp = predictions[0]["systole"][0]
    im1[0].set_ydata(pp)

def animate(i):
    pp = predictions[0]["systole"][i]
    fig.suptitle("power %f"%float(i))
    im1[0].set_ydata(pp)
    return im1

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50)
#anim.save('my_animation.mp4')
plt.show()