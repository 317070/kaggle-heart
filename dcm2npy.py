import dicom
import os
import numpy as np
import glob
np.random.seed(1234)
import matplotlib.pyplot as plt
from matplotlib import animation


"""
total_file_list = [item for i in xrange(1, 501) for item in glob.glob('/home/oncilladock/storage/data/dsb15/train/%d/study/*/*.dcm' % i)]
total_file_list.sort()
print len(total_file_list)
"""

folder_list = glob.glob('/home/oncilladock/storage/data/dsb15/validate/%d/study/*/' % 663)
folder_list.sort()


for folder in folder_list:
    print folder
    file_list = glob.glob('%s/*.dcm' % folder)
    file_list.sort()
    d = dicom.read_file(file_list[0])
    img = d.pixel_array.astype('int')

    fig = plt.figure()
    plt.suptitle(folder)
    im = fig.gca().imshow(img, cmap='gist_gray_r', vmin=0, vmax=255)

    def init():
        im.set_data(img)

    def animate(i):
        d = dicom.read_file(file_list[i])
        img = d.pixel_array.astype('int')
        im.set_data(img)
        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(file_list), interval=50)

plt.show()


"""
fig, plt_axes = plt.subplots(5, len(folder_list)//5+1, sharey=True)
plt_axes = [item for i in plt_axes for item in i]

for i, axis in enumerate(folder_list):
    file_list = glob.glob('%s/*.dcm' % axis)

    d = dicom.read_file(file_list[0])
    img = d.pixel_array.astype('int')
    plt_axes[i] = plt_axes[i].imshow(img, cmap='gist_gray_r', vmin=0, vmax=255)

def animate(i):
    for j, axis in enumerate(folder_list):
        file_list = glob.glob('%s/*.dcm' % axis)
        d = dicom.read_file(file_list[i])
        img = d.pixel_array.astype('int')
        plt_axes[j].set_data(img)
"""