
import re
import data as data_test
import utils
import glob

slice2roi_test_new = utils.load_pkl('../pkl_validate_slice2roi_new.pkl')
print slice2roi_test_new.keys()
print len(slice2roi_test_new.keys())


def get_patient_data(patient_data_path):
    patient_data = []
    spaths = sorted(glob.glob(patient_data_path + '/sax_*.pkl'),
                    key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    pid = re.search(r'/(\d+)/study$', patient_data_path).group(1)
    for s in spaths:
        slice_id = re.search(r'/(sax_\d+\.pkl)$', s).group(1)
        metadata = data_test.read_metadata(s)
        data = data_test.read_slice(s)
        patient_data.append({'data': data, 'metadata': metadata,
                             'slice_id': slice_id, 'patient_id': pid})
    return patient_data


data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
slice2roi_valid = utils.load_pkl('../pkl_validate_slice2roi.pkl')
slice2roi_train = utils.load_pkl('../pkl_train_slice2roi.pkl')

slice2roi_train_new = utils.load_pkl('../pkl_train_slice2roi_new.pkl')

patient_paths = sorted(glob.glob(data_path + '/*/study'))
for p in patient_paths:
    patient_data = get_patient_data(p)
    sd = patient_data[len(patient_data) / 2]
    slice_id = sd['slice_id']
    pid = sd['patient_id']
    if pid in slice2roi_valid:
        slice2roi = slice2roi_valid
    else:
        slice2roi = slice2roi_train

    roi_center = slice2roi[pid][slice_id]['roi_center']
    roi_radii = slice2roi[pid][slice_id]['roi_radii']

    roi_center_new = slice2roi_train_new[pid][slice_id]['roi_center']
    roi_radii_new = slice2roi_train_new[pid][slice_id]['roi_radii']

    print pid
    assert roi_center == roi_center_new
    assert roi_radii == roi_radii_new
