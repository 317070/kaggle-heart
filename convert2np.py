import os
import numpy as np
import dicom
import cPickle as pickle


def read_dicom(filename):
    d = dicom.read_file(filename)
    img = d.pixel_array
    slice_location = d.SliceLocation
    pixel_spacing = d.PixelSpacing
    slice_thickness = d.SliceThickness
    return np.array(img), slice_location, pixel_spacing, slice_thickness


def convert_study_2np(in_path, out_path):
    subdirs = os.listdir(in_path)

    sax_dirs = [in_path + s for s in subdirs if 'sax' in s]
    convert_view_2np(sax_dirs, out_path, 'sax')

    ch2_dirs = [in_path + s for s in subdirs if '2ch' in s]
    convert_view_2np(ch2_dirs, out_path, '2ch')

    ch4_dirs = [in_path + s for s in subdirs if '4ch' in s]
    convert_view_2np(ch4_dirs, out_path, '4ch')


def convert_view_2np(in_paths, out_path, view):
    in_paths.sort(key=lambda x: int(x.split('_')[-1]))
    slices = [int(s.split('_')[-1]) for s in in_paths]
    offsets = []

    # final data is a 4d array: (n_slices, n_times, img_height, img_width)
    d_slices = []  # list of slices (for sax)
    for s in in_paths:
        n_trials = 1  # for most of the studies
        files = os.listdir(s)
        files.sort(key=lambda x: int(x.split('-')[2].split('.')[0]))  # sort by time

        # check if there are multiple trials
        if len(files[0].split('-')) == 4:
            files.sort(key=lambda x: int(x.split('-')[3].split('.')[0]))  # sort by trial
            n_trials = int(files[-1].split('-')[3].split('.')[0])
        n_trials -= 1  # for indexing

        offset, time = [], []  # within one slice offsets should be equal
        d_time = []  # list of 30 images from one cardiac cycle
        for f in files[n_trials * 30:(n_trials + 1) * 30]:  # take the last trial
            offset.append(int(f.split('-')[1]))
            time.append(int(f.split('-')[2].split('.')[0]))
            img = read_dicom(s + '/' + f)[0]
            d_time.append(img)
        assert all(x == y for x, y in zip(time, xrange(1, 31)))  # check time (1-30)
        assert all(x == offset[0] for x in offset)  # check offsets (equal)
        offsets.append(offset[0])
        d_slices.append(d_time)
    # check shapes
    shapes = [dt.shape for ds in d_slices for dt in ds]
    n_shapes = len(set(shapes))
    if n_shapes == 1:
        data = np.array(d_slices, dtype='float32')
    else:
        # raise ValueError('dimension mismatch')
        # TODO: transpose when possible
        # TODO: rescale when needed
        print 'image shapes mismatch'
        print 'study', in_paths[0]
        print set(shapes)
        print shapes
        print '====================================================='
    print 'view:', view
    print 'slices:', slices
    print 'offsets', offsets
    print 'data shape', data.shape
    metadata = None  # TODO: add slicing info, offsets and metadata from dicom files

    out_filename = out_path + view + '.pkl'
    with open(out_filename, 'wb') as f:
        pickle.dump({'data': data,
                     'slices': slices,
                     'offsets': offsets,
                     'metadata': metadata}, f)
    print 'saved to %s' % out_filename
    print


if __name__ == '__main__':
    global_path = '/mnt/sda3/data/kaggle-heart/'
    in_data_path = global_path + 'validate/'
    out_data_path = global_path + 'proc_validate/'

    in_study_paths = os.listdir(in_data_path)
    out_study_paths = [out_data_path + s + '/study/' for s in in_study_paths]
    in_study_paths = [in_data_path + s + '/study/' for s in in_study_paths]

    for p in out_study_paths:
        if not os.path.exists(p):
            os.makedirs(p)

    for s_in, s_out in zip(in_study_paths, out_study_paths):
        print '******** %s *********' % s_in
        convert_study_2np(s_in, s_out)
