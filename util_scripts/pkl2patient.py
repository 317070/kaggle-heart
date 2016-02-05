"""
Converts dicom images from cardiac cycle into 3d numpy arrays (time x height x width) + metadata
and saves into pkl files
"""
import os
import numpy as np
import dicom
import cPickle as pickle
import sys



def clean_metadata(metadatadict):
    for key, value in metadatadict:
        if key == 'PatientAge':
            return int(value[:-1])
    return metadatadict

def read_dicom(filename):
    d = dicom.read_file(filename)
    metadata = {}
    for attr in dir(d):
        if attr[0].isupper() and attr != 'PixelData':
            try:
                metadata[attr] = getattr(d, attr)
            except AttributeError:
                pass
    return np.array(d.pixel_array), metadata


def convert_pkls_2_patient(in_path, out_path):
    pickles = os.listdir(in_path)

    datadict = dict()
    print pickles
    for file in pickles:
        if "sax" in file:
            all_data = pickle.load(open(in_path+file,"r"))
            #d1 = all_data['data']
            d2 = all_data['metadata'][0]
            image_orientation = [float(i) for i in d2["ImageOrientationPatient"]]
            image_position = [float(i) for i in d2["ImagePositionPatient"]]
            pixel_spacing = [float(i) for i in d2["PixelSpacing"]]
            #assert d1.shape[1]==int(d2["Rows"]), (d1.shape[1], d2["Rows"])
            #assert d1.shape[2]==int(d2["Columns"]), (d1.shape[2], d2["Columns"])
            datadict[in_path+file] = {
                "orientation": image_orientation,
                "position": image_position,
                "pixel_spacing": pixel_spacing,
                "rows": int(d2["Rows"]),
                "columns": int(d2["Columns"]),
            }

    for key, data in datadict.iteritems():
        # calculate value of middle pixel
        F = np.array(data["orientation"]).reshape( (2,3) )
        pixel_spacing = data["pixel_spacing"]
        i,j = data["columns"] / 2.0, data["rows"] / 2.0  # reversed order, as per http://nipy.org/nibabel/dicom/dicom_orientation.html
        im_pos = np.array([[i*pixel_spacing[0],j*pixel_spacing[1]]],dtype='float32')
        pos = np.array(data["position"]).reshape((1,3))
        position = np.dot(im_pos, F) + pos
        print position
        data["middle_pixel_position"] = position[0,:]

    # find the keys of the 2 points furthest away from each other
    max_dist = -1.0
    max_dist_keys = []
    for key1, data1 in datadict.iteritems():
        for key2, data2 in datadict.iteritems():
            if key1==key2:
                continue
            p1 = data1["middle_pixel_position"]
            p2 = data2["middle_pixel_position"]
            distance = np.sqrt(np.sum((p1-p2)**2))
            if distance>max_dist:
                max_dist_keys = [key1, key2]
                max_dist = distance


    # project the others on the line between these 2 points
    # sort the keys, so the order is more or less the same as they were
    max_dist_keys.sort()
    print max_dist
    p_ref1 = datadict[max_dist_keys[0]]["middle_pixel_position"]
    p_ref2 = datadict[max_dist_keys[1]]["middle_pixel_position"]
    v1 = p_ref2-p_ref1
    v1 = v1 / np.linalg.norm(v1)
    for key, data in datadict.iteritems():
        v2 = data["middle_pixel_position"]-p_ref1
        scalar = np.inner(v1, v2)
        data["relative_position"] = scalar

    # sort keys and their relative position
    l = [(key, data["relative_position"]) for key, data in datadict.iteritems()]
    l = sorted(l, key=lambda folder: folder[1])

    relative_positions = []
    theoretical_image_spacing = []
    slice_thickness = []
    image_pixel_spacing = []
    image_position = []
    image_orientation = []

    tensor_data = []
    for key, relative_position in l:
        all_data = pickle.load(open(key,"r"))
        d1 = all_data['data']
        tensor_data.append(d1)
        relative_positions.append(relative_position)
        metadata = clean_metadata(all_data['metadata'][0])
        theoretical_image_spacing.append(metadata['SpacingBetweenSlices'])
        slice_thickness.append(metadata['SliceThickness'])
        image_pixel_spacing.append(metadata["PixelSpacing"])
        image_position.append(metadata["ImagePositionPatient"])
        image_orientation.append(metadata["ImageOrientationPatient"])

    # TODO: load those 2 other slices as well
    # TODO: convert to numpy arrays
    # TODO: clean up metadata
    # TODO: write the output

    out_filename = out_path + 'patient.pkl'
    with open(out_filename, 'wb') as f:
        pickle.dump({'data': data,
                     'metadata': metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print 'saved to %s' % out_filename








if __name__ == '__main__':

    convert_pkls_2_patient("/home/oncilladock/444/study/", "/home/oncilladock/444_pkl.py")
    # global_path = '/mnt/sda3/data/kaggle-heart/'
    # dataset = 'validate'
    """
    if len(sys.argv) < 3:
        sys.exit("Usage: dicom2npy.py <global_data_path> <train/validate>")

    global_path = sys.argv[1]
    dataset = sys.argv[2]

    in_data_path = global_path + dataset + '/'
    out_data_path = global_path + 'pkl_' + dataset + '/'

    in_study_paths = os.listdir(in_data_path)
    out_study_paths = [out_data_path + s + '/study/' for s in in_study_paths]
    in_study_paths = [in_data_path + s + '/study/' for s in in_study_paths]

    for p in out_study_paths:
        if not os.path.exists(p):
            os.makedirs(p)

    # s_in = '/mnt/sda3/data/kaggle-heart/validate/643/study/'
    # s_out = '/mnt/sda3/data/kaggle-heart/proc_validate/643/study/'
    # convert_study_2np(s_in, s_out)

    for s_in, s_out in zip(in_study_paths, out_study_paths):
        print '\n******** %s *********' % s_in
        convert_study_2np(s_in, s_out)
    """