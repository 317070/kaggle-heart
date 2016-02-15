import dicom
import os
import numpy as np
import glob

np.random.seed(1234)
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import cPickle as pickle
from pkl2patient import clean_metadata

file_list = glob.glob( os.path.expanduser('~/test/*/*/*.pkl') )
#folder_list = glob.glob( os.path.expanduser('~/storage/data/dsb15/*/*/study/*/') )

print len(file_list)
#folder_list.sort()
np.random.seed(317070)
np.random.shuffle(file_list)



def clean_image_data(imdata, metadata):
    """
    clean up 4d-tensor of imdata consistently (fix contrast, move upside up, etc...)
    :param imdata:
    :return:
    """

    # normalize contrast
    flat_data = np.concatenate([i.flatten() for i in imdata]).flatten()
    high = np.percentile(flat_data, 95.0)
    low  = np.percentile(flat_data, 5.0)
    print high,low
    for i in xrange(len(imdata)):
        image = imdata[i]
        image = 1.0 * (image - low) / (high - low)
        image = np.clip(image, 0.0, 1.0)
        imdata[i] = image

    return imdata


for file in file_list:
    if "sax" not in file:
        continue
    data = pickle.load(open(file, "r"))

    imdata = data['data']

    clean_imdata = clean_image_data([imdata], clean_metadata(data['metadata'][0]))[0]


    print "max:", np.max(imdata)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(50, 100, 600, 300)

    im1 = ax1.imshow(imdata[0], cmap='gist_gray_r', vmin=0, vmax=255)
    im2 = ax2.imshow(clean_imdata[0], cmap='gist_gray_r', vmin=0, vmax=1.0)
    fig.suptitle(file)

    def init():
        im1.set_data(imdata[0])
        im2.set_data(clean_imdata[0])

    def animate(i):
        im1.set_data(imdata[i])
        im2.set_data(clean_imdata[i])
        return im1, im2

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(imdata), interval=50)

    plt.show()