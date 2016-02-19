import Joni_get_data as gd
from extract_roi import axis_likelyhood_surface, sort_images
from glob import glob
import cPickle

train_data_path='/home/lio/geit/data/dsb15_pkl/pkl_train/'
validate_data_path='/home/lio/geit/data/dsb15_pkl/pkl_validate/'

kernel_width = 5
center_margin = 8
num_peaks = 10
num_circles = 20
upscale = 1.5
minradius = 10
maxradius = 40
radstep = 2


def main(data_path):
    paths = glob(data_path+"*")
    assert len(paths)==500 or len(paths) == 200

    ROIinfos = []

    for path in paths:
        data=gd.get_data(path)
        processdata = sort_images(data)
        ROIinfo = {"center":[], "radii":[]}


        lsurface, ROImask, ROIinfo["center"], ROIinfo["radii"] = \
            axis_likelyhood_surface(processdata,
                kernel_width = kernel_width,
                center_margin = center_margin,
                num_peaks = num_peaks,
                num_circles = num_circles,
                upscale = upscale,
                minradius = minradius,
                maxradius = maxradius,
                radstep = radstep)

        print ROIinfo
        ROIinfos.append(ROIinfo)

    return ROIinfos

if __name__ == '__main__':
    trainROIinfos = main(train_data_path)
    validROIinfos = main(validate_data_path)

    cPickle.dump({"train":trainROIinfos, "valid":validROIinfos}, open("ROI_per_stack.pkl","w"))
