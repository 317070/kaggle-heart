import dicom
import os
from dicom.sequence import Sequence
import numpy as np
import glob

np.random.seed(1234)
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import cPickle as pickle

#file_list = glob.glob('/data/dsb15_pkl/pkl_validate/*/study/sax_*.pkl')
file_list = glob.glob(os.path.expanduser('~/test/*/study/sax_*.pkl') )
#folder_list = glob.glob( os.path.expanduser('~/storage/data/dsb15/*/*/study/*/') )

print len(file_list)
#folder_list.sort()
np.random.seed(317070)
np.random.shuffle(file_list)
file_list = sorted(file_list)



def convert_to_number(value):
    value = str(value)
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except:
        pass
    return value



def clean_metadata(metadatadict):
    keys = sorted(list(metadatadict.keys()))
    for key in keys:
        value = metadatadict[key]
        if key == 'PatientAge':
            metadatadict[key] = int(value[:-1])
        else:
            if isinstance(value, Sequence):
                #convert to list
                value = [i for i in value]
            if isinstance(value, (list,)):
                metadatadict[key] = [convert_to_number(i) for i in value]
            else:
                metadatadict[key] = convert_to_number(value)
    return metadatadict


def normalize_contrast(imdata, metadata=None):
    # normalize contrast
    flat_data = np.concatenate([i.flatten() for i in imdata]).flatten()
    high = np.percentile(flat_data, 95.0)
    low  = np.percentile(flat_data, 5.0)
    for i in xrange(len(imdata)):
        image = imdata[i]
        image = 1.0 * (image - low) / (high - low)
        image = np.clip(image, 0.0, 1.0)
        imdata[i] = image

    return imdata


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


def set_upside_up(imdata, metadata=None):
    # turn upside up
    F = np.array(metadata["ImageOrientationPatient"]).reshape( (2,3) )

    f_1 = F[1,:]/np.linalg.norm(F[1,:])
    f_2 = F[0,:]/np.linalg.norm(F[0,:])

    x_e = np.array([1,0,0])
    y_e = np.array([0,1,0])
    z_e = np.array([0,0,1])

    a, b, c = False, False, False
    if abs(np.dot(y_e, f_1)) >= abs(np.dot(y_e, f_2)):
        for i in xrange(len(imdata)):
            image = imdata[i]
            image = np.swapaxes(image, 1, 2)
            imdata[i] = image
            f_1,f_2 = f_2,f_1
        a = True

    if np.dot(y_e, f_1) < 0:
        for i in xrange(len(imdata)):
            imdata[i] = imdata[i][:,::-1,:]
        b = True

    if np.dot(x_e, f_2) < 0:
        for i in xrange(len(imdata)):
            #imdata[i] = imdata[i][:,:,::-1]
            pass
        print f_1, f_2
        print "FLIP COLS %d"%metadata["PatientID"], a,b,True
        col_flipa = True
    else:
        col_flipa = False

    return imdata, col_flipa



def clean_images(data, metadata):
    """
    clean up 4d-tensor of imdata consistently (fix contrast, move upside up, etc...)
    :param data:
    :return:
    """
    from configuration import config
    for process in [set_upside_up, normalize_contrast]:
        data = process(data, metadata)
    return data

for file in file_list:
    if "sax" not in file:
        continue
    data = pickle.load(open(file, "r"))

    imdata = data['data']
    clean_imdata, col_flip = set_upside_up([imdata], clean_metadata(data['metadata'][0]))

    if not col_flip:
        continue

    clean_imdata = clean_imdata[0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(50, 100, 600, 300)

    im1 = ax1.imshow(imdata[0], cmap='gist_gray_r', vmin=0, vmax=255)
    im2 = ax2.imshow(clean_imdata[0], cmap='gist_gray_r', vmin=0, vmax=255)
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