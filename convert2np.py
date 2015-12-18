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


def convert_study_2np(in_path, out_path):
    subdirs = os.listdir(in_path)

    sax_dirs = [in_path + s for s in subdirs if 'sax' in s]
    convert_view_2np(sax_dirs, out_path, 'sax')

    ch2_dirs = [in_path + s for s in subdirs if '2ch' in s]
    convert_view_2np(ch2_dirs, out_path, '2ch')

    ch4_dirs = [in_path + s for s in subdirs if '4ch' in s]
    convert_view_2np(ch4_dirs, out_path, '4ch')


def split_by_orientation(orientations_set, d_slices, m_slices, offsets, slices):
    d, m, s, o = [], [], [], []
    for orientation in orientations_set:
        d_slices1, m_slices1, offsets1, slices1 = [], [], [], []
        for ds, ms, ofs, sl in zip(d_slices, m_slices, offsets, slices):
            # print sl
            # print ds[0].shape
            # print ms[0]['ImageOrientationPatient']
            # print ms[0]['ImagePositionPatient']
            if all(tuple(mt['ImageOrientationPatient']) == orientation for mt in ms):
                d_slices1.append(ds)
                m_slices1.append(ms)
                offsets1.append(ofs)
                slices1.append(sl)

        d.append(np.array(d_slices1))
        m.append(m_slices1)
        s.append(slices1)
        o.append(offsets1)
    return d, m, s, o


def split_by_shape(shapes_set, d_slices, m_slices, offsets, slices):
    d, m, s, o = [], [], [], []
    for shape in shapes_set:
        print '*shape:', shape
        d_slices1, m_slices1, offsets1, slices1 = [], [], [], []
        for ds, ms, ofs, sl in zip(d_slices, m_slices, offsets, slices):
            if all(dt.shape == shape for dt in ds):
                for dt in ds:
                    print dt.shape
                print type(ds), len(ds), ds.shape, ds[0].shape
                print np.asarray(ds).shape
                d_slices1.append(np.array(ds))
                m_slices1.append(ms)
                offsets1.append(ofs)
                slices1.append(sl)
        if slices1:
            d.append(np.array(d_slices1))
            #print np.array(d_slices1).shape
            m.append(m_slices1)
            s.append(slices1)
            o.append(offsets1)
    return d, m, s, o


def convert_view_2np(in_paths, out_path, view):
    in_paths.sort(key=lambda x: int(x.split('_')[-1]))
    slices = [int(s.split('_')[-1]) for s in in_paths]

    offsets = []
    m_slices = []  # metadata per slice
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
        m_slices.append(m_time)

    # check shapes
    img_shapes = [dt.shape for ds in d_slices for dt in ds]
    img_orientations = [tuple(mt['ImageOrientationPatient']) for ms in m_slices for mt in ms]
    shapes_set = list(set(img_shapes))
    orientations_set = list(set(img_orientations))
    d, m, s, o = split_by_orientation(orientations_set, d_slices, m_slices, offsets, slices)
    i = 0
    for dd, mm, ss, oo in zip(d, m, s, o):
        d1, m1, s1, o1 = split_by_shape(shapes_set, dd, mm, oo, ss)
        for dd1, mm1, ss1, oo1 in zip(d1, m1, s1, o1):
            print 'DATA:', dd1.shape
            save_data(dd1, mm1, ss1, oo1, view, out_path, '_' + str(i))
            i += 1


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

    s_in = '/mnt/sda3/data/kaggle-heart/validate/643/study/'
    s_out = '/mnt/sda3/data/kaggle-heart/proc_validate/643/study/'
    convert_study_2np(s_in, s_out)

    for s_in, s_out in zip(in_study_paths, out_study_paths):
        print '******** %s *********' % s_in
        convert_study_2np(s_in, s_out)
