import os
import numpy as np
import dicom
import cPickle as pickle
import sys
from collections import defaultdict


def pad_with_zeros(img, out_shape):
    """pad image with zeros"""
    in_shape = img.shape
    out_img = img
    # rows are equal -> add columns
    col_diff = out_shape[1] - in_shape[1]
    if col_diff:
        pad_img = np.zeros((in_shape[0], col_diff / 2))
        out_img = np.concatenate((pad_img, out_img, pad_img), axis=1)

    row_diff = out_shape[0] - in_shape[0]
    if row_diff:
        pad_img = np.zeros((row_diff / 2, in_shape[1]))
        out_img = np.concatenate((pad_img, out_img, pad_img))

    assert out_shape == out_img.shape
    return out_img


def read_dicom(filename):
    d = dicom.read_file(filename)
    metadata = {}
    for atrr in dir(d):
        if atrr[0].isupper():  # TODO how to check it properly?
            try:
                metadata[atrr] = getattr(d, atrr)
            except AttributeError:
                pass
    return np.array(d.pixel_array), metadata


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
    metadata_slices = []  # metadata per slice
    d_slices = []  # list of data slices (for sax)
    # final data will be 4d array: (n_slices, n_times, img_height, img_width)

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
        m_time = []
        d_time = []  # list of 30 images from one cardiac cycle
        for f in files[n_trials * 30:(n_trials + 1) * 30]:  # take the last trial
            offset.append(int(f.split('-')[1]))
            time.append(int(f.split('-')[2].split('.')[0]))
            img, metadata = read_dicom(s + '/' + f)
            d_time.append(img)
            m_time.append(metadata)
        assert all(x == y for x, y in zip(time, xrange(1, 31)))  # check time (1-30)
        assert all(x == offset[0] for x in offset)  # check offsets (equal)
        offsets.append(offset[0])
        d_slices.append(d_time)
        metadata_slices.append(m_time)
    # check shapes
    img_shapes = [dt.shape for ds in d_slices for dt in ds]
    shapes_set = list(set(img_shapes))
    n_shapes = len(shapes_set)
    if n_shapes == 2:
        print 'shapes are different:'
        print shapes_set
        # check if it's transposed
        if shapes_set[0][0] == shapes_set[1][1] and shapes_set[0][1] == shapes_set[1][0]:
            # select most common shape
            d = defaultdict(int)
            for s in img_shapes:
                d[s] += 1
            common_shape = sorted(d.keys(), key=d.get)[-1]
            for i, ds in enumerate(d_slices):
                for j, dt in enumerate(ds):
                    if dt.shape != common_shape:
                        d_slices[i][j] = np.transpose(dt)
            print 'converted to shape', common_shape, d
        # or we need to pad zeros
        else:
            shapes_sorted = sorted(sorted(shapes_set, key=lambda tup: tup[0]), key=lambda tup: tup[1])
            largest_shape = shapes_sorted[-1]
            for i, ds in enumerate(d_slices):
                for j, dt in enumerate(ds):
                    if dt.shape != largest_shape:
                        d_slices[i][j] = pad_with_zeros(dt, largest_shape)
            print 'converted to shape', largest_shape
    elif n_shapes > 2:
        raise ValueError('3 different img shapes!!!!')

    data = np.array(d_slices)
    print
    print 'view:', view
    print 'slices:', slices
    print 'offsets:', offsets
    print 'data shape:', data.shape
    out_filename = out_path + view + '.pkl'
    with open(out_filename, 'wb') as f:
        pickle.dump({'data': data,
                     'slices': slices,
                     'offsets': offsets,
                     'metadata': metadata_slices}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print 'saved to %s' % out_filename
    print


if __name__ == '__main__':
    if len(sys.argv) == 2:
        global_path = sys.argv[1]
    else:
        global_path = '/mnt/sda3/data/kaggle-heart/'
    in_data_path = global_path + 'validate/'
    out_data_path = global_path + 'proc_validate/'

    in_study_paths = os.listdir(in_data_path)
    out_study_paths = [out_data_path + s + '/study/' for s in in_study_paths]
    in_study_paths = [in_data_path + s + '/study/' for s in in_study_paths]

    for p in out_study_paths:
        if not os.path.exists(p):
            os.makedirs(p)

    # s = '/mnt/sda3/data/kaggle-heart/validate/643/study/'
    # convert_study_2np(s, '/mnt/sda3/data/kaggle-heart/proc_validate/663/study/')

    for s_in, s_out in zip(in_study_paths, out_study_paths):
        print '******** %s *********' % s_in
        convert_study_2np(s_in, s_out)
