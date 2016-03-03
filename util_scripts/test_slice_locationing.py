import numpy as np



import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def slice_location_finder(metadata_dict):
    """
    :param metadata_dict: dict with arbitrary keys, and metadata values
    :return: dict with "relative_position" and "middle_pixel_position" (and others)
    """
    datadict = dict()

    for key, metadata in metadata_dict.iteritems():
        #d1 = all_data['data']
        d2 = metadata
        image_orientation = [float(i) for i in metadata["ImageOrientationPatient"]]
        image_position = [float(i) for i in metadata["ImagePositionPatient"]]
        pixel_spacing = [float(i) for i in metadata["PixelSpacing"]]
        datadict[key] = {
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
        data["middle_pixel_position"] = position[0,:]

    # find the keys of the 2 points furthest away from each other
    if len(datadict)<=1:
        for key, data in datadict.iteritems():
            data["relative_position"] = 0.0
    else:
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
        max_dist_keys.sort(key=natural_keys)
        p_ref1 = datadict[max_dist_keys[0]]["middle_pixel_position"]
        p_ref2 = datadict[max_dist_keys[1]]["middle_pixel_position"]
        v1 = p_ref2-p_ref1
        v1 = v1 / np.linalg.norm(v1)
        for key, data in datadict.iteritems():
            v2 = data["middle_pixel_position"]-p_ref1
            scalar = np.inner(v1, v2)
            data["relative_position"] = scalar

    sorted_indices = [key for key in sorted(datadict.iterkeys(), key=lambda x: datadict[x]["relative_position"])]

    sorted_distances = []
    for i in xrange(len(sorted_indices)-1):
        res = []
        for d1, d2 in [(datadict[sorted_indices[i]], datadict[sorted_indices[i+1]]),
                       (datadict[sorted_indices[i+1]], datadict[sorted_indices[i]])]:
            F = np.array(d1["orientation"]).reshape( (2,3) )
            n = np.cross(F[0,:], F[1,:])
            n = n/np.sqrt(np.sum(n*n))
            p = d2["middle_pixel_position"] - d1["position"]
            distance = np.abs(np.sum(n*p))
            res.append(distance)
        sorted_distances.append(np.mean(res))

    #print sorted_distances

    return datadict, sorted_indices, sorted_distances



if __name__ == '__main__':
    import glob, os
    import cPickle as pickle
    folder_list = glob.glob(os.path.expanduser('~/storage/data/dsb15_pkl/pkl_validate/561/') )

    folder_list = sorted(folder_list)

    # s_in = '/mnt/sda3/data/kaggle-heart/validate/643/study/'
    # s_out = '/mnt/sda3/data/kaggle-heart/proc_validate/643/study/'
    # convert_study_2np(s_in, s_out)

    for patient_folder in folder_list:
        print '\n******** %s *********' % patient_folder.split('/')[-2]
        file_list = glob.glob(patient_folder + 'study/sax_*.pkl')
        metadict = dict()
        for file in file_list:
            metadict[file] = pickle.load(open(file, "r"))['metadata'][0]

        result, sorted_indices, sorted_distances = slice_location_finder(metadict)

        for key in sorted(result.keys()):
            print key.split('/')[-1], result[key]["relative_position"]

        print sorted_indices
        print sorted_distances

