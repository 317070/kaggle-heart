"""
Groups images from different slices into 4d numpy arrays (slice x time x height x width)
and saves into pkl files with extra metadata
"""

import os
import numpy as np
import cPickle as pickle
import sys
import shutil


def convert_study(in_path, out_path, sax_groups):
    study_files = os.listdir(in_path)

    sax_files = [in_path + sf for sf in study_files if 'sax' in sf]
    group_sax(sax_files, out_path, sax_groups)

    for sf in study_files:
        if '2ch' in sf or '4ch' in sf:
            shutil.copyfile(in_path + sf, out_path + sf)


def group_sax(in_paths, out_path, groups):
    in_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    slices = np.array([int(p.split('_')[-1].split('.')[0]) for p in in_paths])

    for i, group in enumerate(groups):
        data, metadata = [], []
        for g in group:
            idx = np.where(slices == g)[0][0]
            with open(in_paths[idx], 'rb') as f:
                d = pickle.load(f)
            data.append(d['data'])
            metadata.append(d['metadata'])

        out_filename = out_path + 'sax_g%s.pkl' % i
        with open(out_filename, 'wb') as f:
            pickle.dump({'data': np.array(data), 'metadata': metadata, 'slices': group}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print 'group %s saved to %s' % (group, out_filename)


if __name__ == '__main__':
    # global_path = '/mnt/sda3/data/kaggle-heart/'
    # dataset = 'proc_validate'

    if len(sys.argv) < 3:
        sys.exit("Usage: dicom2npy.py <global_data_path> <pkl_train/pkl_validate>")

    global_path = sys.argv[1]
    dataset = sys.argv[2]

    in_data_path = global_path + dataset + '/'
    out_data_path = global_path + '4d_' + dataset + '/'

    in_study_paths = os.listdir(in_data_path)
    out_study_paths = [out_data_path + s + '/study/' for s in in_study_paths]
    in_study_paths = [in_data_path + s + '/study/' for s in in_study_paths]
    groups = [[7, 8], [9, 10, 11, 12, 13, 14, 15, 20], [17]]  # TODO read groups

    for p in out_study_paths:
        if not os.path.exists(p):
            os.makedirs(p)

    # s_in = '/mnt/sda3/data/kaggle-heart/proc_validate/643/study/'
    # s_out = '/mnt/sda3/data/kaggle-heart/4d_proc_validate/643/study/'
    # if not os.path.exists(s_out):
    #     os.makedirs(s_out)
    # convert_study(s_in, s_out, groups)

    for s_in, s_out, g in zip(in_study_paths, out_study_paths, groups):
        print '******** %s *********' % s_in
        convert_study(s_in, s_out, g)
