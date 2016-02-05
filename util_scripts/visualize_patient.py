import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import glob
import os
from mpl_toolkits.mplot3d import Axes3D

print "Looking for the pickle files..."
files = sorted(glob.glob(os.path.expanduser("~/444/study/*.pkl")))
print "Loading data..."

data = []
for file in files:
    all_data = pickle.load(open(file,"r"))
    d1 = all_data['data']
    d2 = all_data['metadata'][0]  # assume constant #TODO: check this!
    image_orientation = [float(i) for i in d2["ImageOrientationPatient"]]
    image_position = [float(i) for i in d2["ImagePositionPatient"]]
    pixel_spacing = [float(i) for i in d2["PixelSpacing"]]
    data.append( (d1,image_orientation,image_position, pixel_spacing) )
    assert d1.shape[1]==int(d2["Rows"]), (d1.shape[1], d2["Rows"])
    assert d1.shape[2]==int(d2["Columns"]), (d1.shape[2], d2["Columns"])



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='equal')
# store points in here
x_points = []
y_points = []
z_points = []
s_points = []

RESOLUTION = 4

for slice, image_orientation, image_position, pixel_spacing in data:
    image = slice[0]
    for j in xrange(0,image.shape[0],RESOLUTION):
        for i in xrange(0,image.shape[1],RESOLUTION):
            F = np.array(image_orientation).reshape( (2,3) )
            im_pos = np.array([[i*pixel_spacing[0],j*pixel_spacing[1]]],dtype='float32')
            pos = np.array(image_position).reshape((1,3))
            position = np.dot(im_pos, F) + pos
            if image[j,i]>128:
                x_points.append(position[0,0])
                y_points.append(position[0,1])
                z_points.append(position[0,2])
                s_points.append(1)
                #s_points.append(image[j,i])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.scatter(x_points,
         y_points,
         z_points,
         c=s_points,
         cmap="bone",
         marker='.',
         edgecolors='face')
print "plotting"
plt.show()