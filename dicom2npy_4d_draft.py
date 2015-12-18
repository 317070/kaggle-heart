"""
Groups images from different slices into 4d numpy arrays (slice x time x height x width)
and saves into pkl files with extra metadata
"""

import os
import numpy as np
import dicom
import cPickle as pickle
import sys


def read_dicom(filename):
    d = dicom.read_file(filename)
    metadata = {}
    for attr in dir(d):
        if attr[0].isupper():
            try:
                metadata[attr] = getattr(d, attr)
            except AttributeError:
                pass
    return np.array(d.pixel_array), metadata


def save_data(data, metadata, slices, offsets, view, out_path, batch=''):
    print 'view:', view
    print 'slices:', slices
    print 'offsets:', offsets
    print 'data shape:', data.shape
    out_filename = out_path + view + batch + '.pkl'
    with open(out_filename, 'wb') as f:
        pickle.dump({'data': data,
                     'slices': slices,
                     'offsets': offsets,
                     'metadata': metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print 'saved to %s' % out_filename
    print


def convert_study_2np(in_path, out_path, grouping):
    subdirs = os.listdir(in_path)

    sax_dirs = [in_path + s for s in subdirs if 'sax' in s]
    convert_view_2np(sax_dirs, out_path, 'sax', grouping)

    ch2_dirs = [in_path + s for s in subdirs if '2ch' in s]
    convert_view_2np(ch2_dirs, out_path, '2ch')

    ch4_dirs = [in_path + s for s in subdirs if '4ch' in s]
    convert_view_2np(ch4_dirs, out_path, '4ch')


def convert_view_2np(in_paths, out_path, view, grouping=None):
    in_paths.sort(key=lambda x: int(x.split('_')[-1]))

    slices = [int(p.split('_')[-1]) for p in in_paths]
    offsets = []
    m_slices = []  # metadata per slice
    d_slices = []  # list of data slices (for sax)

    # final data will be 4d array: (n_slices, n_times, img_height, img_width)

    for in_path in in_paths:
        pass


if __name__ == '__main__':
    # global_path = '/mnt/sda3/data/kaggle-heart/'
    # dataset = 'pkl_validate'

    if len(sys.argv) < 3:
        sys.exit("Usage: dicom2npy.py <global_data_path> <train/validate>")

    global_path = sys.argv[1]
    dataset = sys.argv[2]

    in_data_path = global_path + dataset + '/'
    out_data_path = global_path + '4d_pkl_' + dataset + '/'

    in_study_paths = os.listdir(in_data_path)
    out_study_paths = [out_data_path + s + '/study/' for s in in_study_paths]
    in_study_paths = [in_data_path + s + '/study/' for s in in_study_paths]
    groups = None

    for p in out_study_paths:
        if not os.path.exists(p):
            os.makedirs(p)

    for s_in, s_out, g in zip(in_study_paths, out_study_paths, groups):
        print '******** %s *********' % s_in
        convert_study_2np(s_in, s_out, g)
