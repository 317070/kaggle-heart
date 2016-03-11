from dicom.sequence import Sequence
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import glob
import os
from mpl_toolkits.mplot3d import Axes3D


def merge_dicts(dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res

def _load_file(path):
    with open(path, "r") as f:
        data = pickle.load(f)
    return data

_HOUGH_ROI_PATHS = (
    os.path.expanduser('/data/dsb15_pkl/pkl_train_slice2roi.pkl'),
    os.path.expanduser('/data/dsb15_pkl/pkl_validate_slice2roi.pkl'),)
_hough_rois = merge_dicts(map(_load_file, _HOUGH_ROI_PATHS))

def _enhance_metadata(metadata, patient_id, slice_name):
    # Add hough roi metadata using relative coordinates
    roi_center = list(_hough_rois[str(patient_id)][slice_name]['roi_center'])
    if not roi_center == (None, None):
        roi_center[0] = float(roi_center[0]) / metadata['Rows']
        roi_center[1] = float(roi_center[1]) / metadata['Columns']
    metadata['hough_roi'] = tuple(roi_center)
    metadata['hough_roi_radii'] = _hough_rois[str(patient_id)][slice_name]['roi_radii']

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




def orthogonal_projection_on_slice(percentual_coordinate, source_metadata, target_metadata):
    point = np.array([[percentual_coordinate[0]],
                      [percentual_coordinate[1]],
                      [0],
                      [1]])
    image_size = [source_metadata["Rows"], source_metadata["Columns"]]
    point = np.dot(np.array(  [[image_size[0],0,0,0],
                               [0,image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = source_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[pixel_spacing[0],0,0,0],
                               [0,pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    Fa = np.array(source_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    posa = source_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[Fa[0,0],Fa[1,0],0,posa[0]],
                               [Fa[0,1],Fa[1,1],0,posa[1]],
                               [Fa[0,2],Fa[1,2],0,posa[2]],
                               [0,0,0,1]]), point)
    posb = target_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[1,0,0,-posb[0]],
                               [0,1,0,-posb[1]],
                               [0,0,1,-posb[2]],
                               [0,0,0,1]]), point)
    Fb = np.array(target_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    ff0 = np.sqrt(np.sum(Fb[0,:]*Fb[0,:]))
    ff1 = np.sqrt(np.sum(Fb[1,:]*Fb[1,:]))

    point = np.dot(np.array(  [[Fb[0,0]/ff0,Fb[0,1]/ff0,Fb[0,2]/ff0,0],
                               [Fb[1,0]/ff1,Fb[1,1]/ff1,Fb[1,2]/ff1,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = target_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[1./pixel_spacing[0],0,0,0],
                               [0,1./pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    image_size = [target_metadata["Rows"], target_metadata["Columns"]]
    point = np.dot(np.array(  [[1./image_size[0],0,0,0],
                               [0,1./image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    return point[:2,0]  # percentual coordinate as well



for patient_id in xrange(246,701):
    print "Looking for the pickle files..."
    files = sorted(glob.glob(os.path.expanduser("/data/dsb15_final/pkl_train/%d/study/*.pkl" % patient_id)))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # to put it into the upper left corner for example:
    f.canvas.manager.resize(*f.canvas.manager.window.maxsize())

    has4ch = len([f for f in files if "4ch" in f]) > 0
    # show 4ch file
    if has4ch:
        ch4_file = [f for f in files if "4ch" in f][0]
        all_data = pickle.load(open(ch4_file,"r"))
        d1 = all_data['data']
        ch4_metadata = clean_metadata(all_data['metadata'][0])
        ax1.imshow(d1[0])
        ax1.set_aspect('equal')
    ch2_file = [f for f in files if "2ch" in f][0]
    all_data = pickle.load(open(ch2_file,"r"))
    d1 = all_data['data']
    ch2_metadata = clean_metadata(all_data['metadata'][0])
    ax2.imshow(d1[0])
    ax2.set_aspect('equal')

    print "Loading data..."
    for file in files:
        if "sax" in file:
            all_data = pickle.load(open(file,"r"))
            d1 = all_data['data']
            metadata = clean_metadata(all_data['metadata'][0])
            _enhance_metadata(metadata, patient_id, slice_name = os.path.basename(file))
            if has4ch:
                ch4_point = orthogonal_projection_on_slice(metadata['hough_roi'], metadata, ch4_metadata)
                data_x = [ch4_point[1] * ch4_metadata['Columns']]
                data_y = [ch4_point[0] * ch4_metadata['Rows']]
                ax1.plot(data_x, data_y, 'o')
            ch2_point = orthogonal_projection_on_slice(metadata['hough_roi'], metadata, ch2_metadata)
            data_x = [ch2_point[1] * ch2_metadata['Columns']]
            data_y = [ch2_point[0] * ch2_metadata['Rows']]
            ax2.plot(data_x, data_y, 'o')


    print file
    all_data = pickle.load(open(file,"r"))
    d1 = all_data['data']
    ax3.imshow(d1[0])
    data_x = [metadata['hough_roi'][1] * metadata['Columns']]
    data_y = [metadata['hough_roi'][0] * metadata['Rows']]
    ax3.plot(data_x, data_y, 'or')

    print "plotting"
    plt.show()
