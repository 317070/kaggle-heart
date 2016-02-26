import glob
import re
import data as data_test
import utils

data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'


def test_1():
    patient_path = sorted(glob.glob(data_path + '/*/study'))
    max_norm_shape_0 = 0
    max_norm_shape_1 = 0
    for p in patient_path:
        spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
        for s in spaths:
            data = data_test.read_slice(s)
            metadata = data_test.read_metadata(s)
            img_shape = data.shape[-2:]
            norm_shape_0 = min(img_shape) * metadata['PixelSpacing'][0]
            norm_shape_1 = max(img_shape) * metadata['PixelSpacing'][1]

            if norm_shape_0 > max_norm_shape_0:
                max_norm_shape_0 = norm_shape_0

            if norm_shape_1 > max_norm_shape_1:
                max_norm_shape_1 = norm_shape_1

    print max_norm_shape_0
    print max_norm_shape_1


def test_3():
    patient_path = sorted(glob.glob(data_path + '/*/study'))
    px = []
    for p in patient_path:
        spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
        for s in spaths:
            # data = data_test.read_slice(s)
            metadata = data_test.read_metadata(s)
            px.append(metadata['SliceThickness'])

    print px
    px = set(px)
    if len(px) != 1:
        print px


def test_2():
    slice2roi = utils.load_pkl('pkl_train_slice2roi.pkl')
    slice2roi_valid = utils.load_pkl('pkl_validate_slice2roi.pkl')
    slice2roi.update(slice2roi_valid)

    patient_path = sorted(glob.glob(data_path + '/*/study'))
    max_d = 0
    max_dm = 0
    max_heart = ''
    max_heartm = ''
    for p in patient_path:
        spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
        for s in spaths:
            pid = utils.get_patient_id(s)
            sid = utils.get_slice_id(s)
            roi = slice2roi[pid][sid]
            diameter = 2 * max(roi['roi_radii'])

            metadata = data_test.read_metadata(s)
            dm = diameter * metadata['PixelSpacing'][0]

            if dm > max_dm:
                max_dm = dm
                max_heartm = s

            if diameter > max_d:
                max_d = diameter
                max_heart = s

    print max_d, max_heart
    print max_dm, max_heartm


def test4():
    patient_path = sorted(glob.glob(data_path + '/*/study'))
    px = []
    for p in patient_path:
        spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
        for s in spaths:
            metadata = data_test.read_metadata(s)
            px.append(metadata['PatientAge'])
    px = set(px)
    if len(px) != 1:
        print px


if __name__ == '__main__':
    test4()
