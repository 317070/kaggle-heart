import json
import utils
import os
import slice2roi

with open('../SETTINGS.json') as data_file:
    paths = json.load(data_file)

SUBMISSION_NR = paths["SUBMISSION_NR"]
METADATA_PATH = paths["METADATA_PATH"]
PREDICTIONS_PATH = paths["PREDICTIONS_PATH"]
SUBMISSION_PATH = paths["SUBMISSION_PATH"]

TRAIN_DATA_PATH = paths["TRAIN_DATA_PATH"]
PKL_TRAIN_DATA_PATH = paths["PKL_TRAIN_DATA_PATH"]
utils.check_data_paths(TRAIN_DATA_PATH, PKL_TRAIN_DATA_PATH)

VALIDATE_DATA_PATH = paths["VALIDATE_DATA_PATH"]
PKL_VALIDATE_DATA_PATH = paths["PKL_VALIDATE_DATA_PATH"]
utils.check_data_paths(VALIDATE_DATA_PATH, PKL_VALIDATE_DATA_PATH)

TRAIN_LABELS_PATH = paths["TRAIN_LABELS_PATH"]
if not os.path.isfile(TRAIN_LABELS_PATH):
    raise ValueError('no file with train labels')

# TODO: next time move this a better place
if not os.path.isfile('pkl_train_slice2roi.pkl'):
    print 'Generating ROI'
    slice2roi.get_slice2roi(PKL_TRAIN_DATA_PATH, 'pkl_train_slice2roi.pkl')

if not os.path.isfile('pkl_validate_slice2roi.pkl'):
    print 'Generating ROI'
    slice2roi.get_slice2roi(PKL_VALIDATE_DATA_PATH, 'pkl_validate_slice2roi.pkl')

if not os.path.isfile('pkl_train_slice2roi_10.pkl'):
    print 'Generating ROI'
    slice2roi.get_slice2roi(PKL_TRAIN_DATA_PATH, 'pkl_train_slice2roi_10.pkl', num_circles=10)

if not os.path.isfile('pkl_validate_slice2roi_10.pkl'):
    print 'Generating ROI'
    slice2roi.get_slice2roi(PKL_VALIDATE_DATA_PATH, 'pkl_validate_slice2roi_10.pkl', num_circles=10)
