import  matplotlib
matplotlib.use('TkAgg')
print matplotlib.get_backend()



import glob
import re
import cPickle as pickle

from matplotlib import animation
import matplotlib.pyplot as plt

pid = 501
slice_id = 0
data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
patient_path = data_path + '/%s/study/' % pid
spaths = sorted(glob.glob(patient_path + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
data = pickle.load(open(spaths[slice_id]))['data']

img = data[0]

def init():
    im.set_data(img)


def animate(i):
    d = data[i]
    img = d.pixel_array.astype('int')
    im.set_data(img)
    return im


fig = plt.figure()
im = fig.gca().imshow(img, cmap='gist_gray_r', vmin=0, vmax=255)
print 'hello'
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30, interval=50)

plt.show()
print 'hello'

