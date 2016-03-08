"""
Converts dicom images from cardiac cycle into 3d numpy arrays (time x height x width) + metadata
and saves into pkl files
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
        if attr[0].isupper() and attr != 'PixelData':
            try:
                metadata[attr] = getattr(d, attr)
            except AttributeError:
                pass
    return np.array(d.pixel_array), metadata


def save_data(data, metadata, in_path, out_path):
    filename = in_path.split('/')[-1]
    out_filename = out_path + filename + '.pkl'
    with open(out_filename, 'wb') as f:
        pickle.dump({'data': data,
                     'metadata': metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print 'saved to %s' % out_filename


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

    # final data will be 3d array: (n_times, img_height, img_width)
    for in_path in in_paths:
        n_trials = 1  # for most of the studies
        files = os.listdir(in_path)
        files.sort(key=lambda x: int(x.split('-')[2].split('.')[0]))  # sort by time

        # check if there are multiple trials
        if len(files[0].split('-')) == 4:
            files.sort(key=lambda x: int(x.split('-')[3].split('.')[0]))  # sort by trial
            n_trials = int(files[-1].split('-')[3].split('.')[0])
        n_trials -= 1  # for indexing

        offsets, times = [], []  # within one slice offsets should be equal
        m_time = []
        d_time = []  # list of 30 images from one cardiac cycle
        for f in files[n_trials * 30:(n_trials + 1) * 30]:  # take the last trial
            offsets.append(int(f.split('-')[1]))
            times.append(int(f.split('-')[2].split('.')[0]))
            img, metadata = read_dicom(in_path + '/' + f)
            d_time.append(img)
            m_time.append(metadata)

        assert all(x == y for x, y in zip(times, xrange(1, 31)))  # check time (1-30)
        assert all(x == offsets[0] for x in offsets)  # check offsets (equal)

        # check shapes and orientations (redundant)
        img_shapes = [dt.shape for dt in d_time]
        shapes_set = list(set(img_shapes))
        assert len(shapes_set) == 1
        img_orientations = [tuple(mt['ImageOrientationPatient']) for mt in m_time]
        orientations_set = list(set(img_orientations))
        assert len(orientations_set) == 1
        save_data(np.array(d_time), m_time, in_path, out_path)


def preprocess(in_data_path, out_data_path=None):
    dataset = in_data_path.split('/')[-1]
    in_data_path = in_data_path
    if not out_data_path:
        out_data_path = in_data_path.replace(dataset, 'pkl_' + dataset)

    in_study_paths = sorted(os.listdir(in_data_path))
    out_study_paths = [out_data_path + '/' + s + '/study/' for s in in_study_paths]
    in_study_paths = [in_data_path + '/' + s + '/study/' for s in in_study_paths]

    for p in out_study_paths:
        if not os.path.exists(p):
            os.makedirs(p)

    for s_in, s_out in zip(in_study_paths, out_study_paths):
        print '\n******** %s *********' % s_in
        convert_study_2np(s_in, s_out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: dicom2npy.py <global_data_path>")
    global_path = sys.argv[1]
    preprocess(global_path)
