import dicom
import os
import numpy as np
import glob
np.random.seed(1234)
import matplotlib.pyplot as plt
from matplotlib import animation
import os

"""
total_file_list = [item for i in xrange(1, 501) for item in glob.glob('/home/oncilladock/storage/data/dsb15/train/%d/study/*/*.dcm' % i)]
total_file_list.sort()
print len(total_file_list)
"""

folder_list = glob.glob( os.path.expanduser('~/storage/data/dsb15/lv-challenge/*/*/') )
#folder_list = glob.glob( os.path.expanduser('~/storage/data/dsb15/*/*/study/*/') )

print len(folder_list)
#folder_list.sort()
np.random.seed(317070)
np.random.shuffle(folder_list)

x = []
y = []
orient = []
posit = []
for folder in folder_list[27:]:
    print folder
    file_list = glob.glob('%s/*.dcm' % folder)
    if not file_list:
        continue
    file_list.sort()

    vmin = 1e10
    vmax = -1e10


    for file in file_list:
        d = dicom.read_file(file)
        #print d.ImageType, d.SoftwareVersions
        #print d.SliceLocation, d.SliceThickness
        #print d.SmallestImagePixelValue, d.LargestImagePixelValue
        #print d.NumberOfPhaseEncodingSteps
        if d.SmallestImagePixelValue < vmin:
            vmin = d.SmallestImagePixelValue
        if d.LargestImagePixelValue > vmax:
            vmax = d.LargestImagePixelValue
        img = d.pixel_array.astype('int')

    """
    posit.append(d.ImagePositionPatient)
    orient.append(d.ImageOrientationPatient)
    x.append(d.ImagePositionPatient[0])
    y.append(d.ImagePositionPatient[1:])
    plt.gca().annotate(folder.split('/')[-2], xy=(x[-1], y[-1][0]), xytext=(x[-1]+20, y[-1][0]+20),
            arrowprops=dict(facecolor='black', shrink=0.05))
    plt.gca().annotate(folder.split('/')[-2], xy=(x[-1], y[-1][1]), xytext=(x[-1]+20, y[-1][1]+20),
            arrowprops=dict(facecolor='black', shrink=0.05))
    print d.ImageOrientationPatient, d.ImagePositionPatient
    """
    #print d.ManufacturerModelName, d.WindowCenterWidthExplanation
    #fig = plt.figure()
    #n, bins, patches = plt.hist(img.flatten(), 50, normed=1, facecolor='green', alpha=0.75)

    fig = plt.figure()

    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(50, 100, 640, 545)

    plt.suptitle(folder)
    vmax = 255
    print "max:",vmax
    im = fig.gca().imshow(img, cmap='gist_gray_r', vmin=vmin, vmax=vmax)

    def init():
        im.set_data(img)

    def animate(i):
        d = dicom.read_file(file_list[i])
        img = d.pixel_array.astype('int')
        im.set_data(img)
        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(file_list), interval=50)

    plt.show()

#print posit, orient
#plt.plot(x,y,'.')
#plt.show()

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