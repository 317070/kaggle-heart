if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')

from dicom.sequence import Sequence
import skimage
from image_transform import fast_warp
from util_scripts.test_slice_locationing import slice_location_finder

import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import glob
import os


def _load_file(path):
    with open(path, "r") as f:
        data = pickle.load(f)
    return data

def merge_dicts(dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res

try:
    _HOUGH_ROI_PATHS = (
        os.path.expanduser('/mnt/storage/data/dsb15_pkl/pkl_train_slice2roi.pkl'),
        os.path.expanduser('/mnt/storage/data/dsb15_pkl/pkl_validate_slice2roi.pkl'),
    )
    _hough_rois = merge_dicts(map(_load_file, _HOUGH_ROI_PATHS))
except:
    pass

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


def patient_coor_from_slice(percentual_coordinate, source_metadata):
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

    return point[:3,0]  # patient coordinate


def point_projection_on_slice(point, target_metadata):
    point = np.array([[point[0]],
                      [point[1]],
                      [point[2]],
                      [1]])
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
    return point[:2,0]  # percentual coordinate as well



def get_chan_transformations(ch2_metadata=None,
                             ch4_metadata=None,
                             top_point_metadata=None,
                             bottom_point_metadata=None,
                             output_width = 100):

    has_both_chans = False
    if ch2_metadata is None and ch4_metadata is None:
        raise "Need at least one of these slices"
    elif ch2_metadata and ch4_metadata is None:
        ch4_metadata = ch2_metadata
    elif ch4_metadata and ch2_metadata is None:
        ch2_metadata = ch4_metadata
    else:
        has_both_chans = True
    #print has_both_chans

    F2 = np.array(ch2_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    F4 = np.array(ch4_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]

    n2 = np.cross(F2[0,:], F2[1,:])
    n4 = np.cross(F4[0,:], F4[1,:])

    b2 = np.sum(n2 * np.array(ch2_metadata["ImagePositionPatient"]))
    b4 = np.sum(n4 * np.array(ch4_metadata["ImagePositionPatient"]))

    # find top and bottom of my view
    top_point = patient_coor_from_slice(top_point_metadata["hough_roi"], top_point_metadata)
    bottom_point = patient_coor_from_slice(bottom_point_metadata["hough_roi"], bottom_point_metadata)

    # if it has both chan's: middle line is the common line!
    if has_both_chans:
        F5 = np.cross(n2, n4)
        A = np.array([n2, n4])
        b = np.array([b2, b4])
        #print A, b
        P, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print P, rnorm, rank, s

        # find top and bottom on the line
        A = np.array([F5]).T
        b = np.array(top_point)
        #print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print sc, rnorm, rank, s

        top_point = sc[0] * F5 + P

        A = np.array([F5]).T
        b = np.array(bottom_point)
        #print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print sc, rnorm, rank, s

        bottom_point = sc[0] * F5 + P

    ## FIND THE affine transformation ch2 needs:

    ch2_top_point = point_projection_on_slice(top_point, ch2_metadata)
    ch2_bottom_point = point_projection_on_slice(bottom_point, ch2_metadata)
    n = np.array([ch2_bottom_point[1] - ch2_top_point[1], ch2_top_point[0] - ch2_bottom_point[0]])
    ch2_third_point = ch2_top_point + n/2
    #print np.sum(n*n)
    if False:#np.sum(n*n)<1 and has_both_chans:
        return get_chan_transformations(ch2_metadata=None,
                                        ch4_metadata=ch4_metadata,
                                        top_point_metadata=top_point_metadata,
                                        bottom_point_metadata=bottom_point_metadata,
                                        output_width = output_width)

    A = np.array([[ch2_top_point[0], ch2_top_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_top_point[0], ch2_top_point[1], 1],
                  [ch2_bottom_point[0], ch2_bottom_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_bottom_point[0], ch2_bottom_point[1], 1],
                  [ch2_third_point[0], ch2_third_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_third_point[0], ch2_third_point[1], 1],])
    b = np.array([0,0.5*output_width,output_width,0.5*output_width,0,0])
    #print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A,b)
    #print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [    0,     0,    1]]))
    ch2_form_fix  = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    # same for ch4
    ch4_top_point = point_projection_on_slice(top_point, ch4_metadata)
    ch4_bottom_point = point_projection_on_slice(bottom_point, ch4_metadata)
    n = np.array([ch4_bottom_point[1] - ch4_top_point[1], ch4_top_point[0] - ch4_bottom_point[0]])
    ch4_third_point = ch4_top_point + n/2
    #print np.sum(n*n)
    if np.sum(n*n)<1 and has_both_chans:
        return get_chan_transformations(ch2_metadata=ch2_metadata,
                                         ch4_metadata=None,
                                         top_point_metadata=top_point_metadata,
                                         bottom_point_metadata=bottom_point_metadata,
                                         output_width = output_width)

    A = np.array([[ch4_top_point[0], ch4_top_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_top_point[0], ch4_top_point[1], 1],
                  [ch4_bottom_point[0], ch4_bottom_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_bottom_point[0], ch4_bottom_point[1], 1],
                  [ch4_third_point[0], ch4_third_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_third_point[0], ch4_third_point[1], 1],])
    b = np.array([0,0.5*output_width,output_width,0.5*output_width,0,0])
    #print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A,b)
    #print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [    0,     0,    1]]))
    ch4_form_fix  = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    return ch2_form_fix, ch4_form_fix




if __name__ == "__main__":

    for patient_id in xrange(611, 612):
        print "Looking for the pickle files..."
        files = sorted(glob.glob(os.path.expanduser("~/storage/data/dsb15_pkl/pkl_validate/%d/study/*.pkl" % patient_id)))

        ch2_file = [f for f in files if "2ch" in f][0]
        if len([f for f in files if "4ch" in f]) > 0:
            has_ch4 = True
            ch4_file = [f for f in files if "4ch" in f][0]
        else:
            has_ch4 = False
            ch4_file = ch2_file

        sax_files = [f for f in files if "sax" in f]
        print "%d sax files" % len(sax_files)

        ch2_metadata = clean_metadata(pickle.load(open(ch2_file))["metadata"][0])
        ch4_metadata = clean_metadata(pickle.load(open(ch4_file))["metadata"][0])

        ch2_data = pickle.load(open(ch2_file))["data"]
        ch4_data = pickle.load(open(ch4_file))["data"]

        metadata_dict = dict()
        for file in files:
            if "sax" in file:
                all_data = pickle.load(open(file,"r"))
                metadata_dict[file] = all_data['metadata'][0]
        datadict, sorted_indices, sorted_distances = slice_location_finder(metadata_dict)

        # find top and bottom of my view

        top_point_enhanced_metadata = datadict[sorted_indices[0]]["middle_pixel_position"]
        bottom_point_enhanced_metadata = datadict[sorted_indices[-1]]["middle_pixel_position"]

        top_point_enhanced_metadata = pickle.load(open(sorted_indices[0],"r"))['metadata'][0]
        _enhance_metadata(top_point_enhanced_metadata, patient_id, slice_name = os.path.basename(sorted_indices[0]))

        bottom_point_enhanced_metadata = pickle.load(open(sorted_indices[-1],"r"))['metadata'][0]
        _enhance_metadata(bottom_point_enhanced_metadata, patient_id, slice_name = os.path.basename(sorted_indices[-1]))

        OUTPUT_SIZE = 100

        trf_2ch, trf_4ch = get_chan_transformations(
            ch2_metadata=ch2_metadata,
            ch4_metadata=ch4_metadata if has_ch4 else None,
            top_point_metadata = top_point_enhanced_metadata,
            bottom_point_metadata = bottom_point_enhanced_metadata,
            output_width=OUTPUT_SIZE
            )



        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # to put it into the upper left corner for example:
        #f.canvas.manager.resize(*f.canvas.manager.window.maxsize())

        ch4_result = fast_warp(ch4_data[0], trf_4ch, output_shape=(OUTPUT_SIZE, OUTPUT_SIZE))
        #ch4_result[50,:] = 0
        ax1.imshow(ch4_result)
        ax1.set_aspect('equal')

        ch2_result = fast_warp(ch2_data[0], trf_2ch, output_shape=(OUTPUT_SIZE, OUTPUT_SIZE))
        #ch2_result[50,:] = 0
        ax2.imshow(ch2_result)
        ax2.set_aspect('equal')

        ax3.imshow(ch4_data[0])
        ax3.set_aspect('equal')

        ax4.imshow(ch2_data[0])
        ax4.set_aspect('equal')

        print "Loading data..."
        for file in files:
            if "sax" in file:
                all_data = pickle.load(open(file,"r"))
                metadata = all_data['metadata'][0]
                d1 = all_data['data']
                _enhance_metadata(metadata, patient_id, slice_name = os.path.basename(file))
                ch4_point = orthogonal_projection_on_slice(metadata['hough_roi'], metadata, ch4_metadata)
                data_x = [ch4_point[1] * ch4_metadata['Columns']]
                data_y = [ch4_point[0] * ch4_metadata['Rows']]
                ax3.plot(data_x, data_y, 'x')
                ch2_point = orthogonal_projection_on_slice(metadata['hough_roi'], metadata, ch2_metadata)
                data_x = [ch2_point[1] * ch2_metadata['Columns']]
                data_y = [ch2_point[0] * ch2_metadata['Rows']]
                ax4.plot(data_x, data_y, 'x')


        print "plotting"
        plt.show()