
import glob
import cPickle as pickle
import os

import numpy as np

import data_loader


PREDICTIONS_PATH = "/mnt/storage/metadata/kaggle-heart/predictions/"

SLICEMODEL_NAMES = (
  'je_ss_*', 'je_single_slice*', 'j2_single_slice*')

PATIENT_MODEL_ORDERED_SLICES_NAMES = (
  'je_meta_*', 'je_os*', 'je_lio_rnn.*')
PATIENT_MODEL_UNORDERED_SLICES_NAMES = (
  'je_rs*', 'je_lio_rnn_360_sort.*')


def get_all_models(names):
    res = []
    for sn in names:
        res += glob.glob(os.path.join(PREDICTIONS_PATH, sn))
    return res


def dump_prediction(prediction_dict, prediction_path):
    print "dumping prediction file to %s" % prediction_path
    with open(prediction_path, 'w') as f:
        pickle.dump(prediction_dict, f, pickle.HIGHEST_PROTOCOL)
    print "prediction file dumped"


def _is_empty_prediction(patient_prediction):
    return (
      len(patient_prediction['systole']) == 0
      and len(patient_prediction['diastole']) == 0
      and 'patient' in patient_prediction
      and len(patient_prediction) == 3)


def _clean_prediction(patient_prediction):
    pid = patient_prediction['patient'] 
    for tag in patient_prediction.keys():
        del patient_prediction[tag]
    patient_prediction['patient'] = pid   
    patient_prediction["systole"] = np.zeros((0,600))
    patient_prediction["diastole"] = np.zeros((0,600))


def fix_metadata(metadata):
    flag_edited = False
    for midx, patient_prediction in enumerate(metadata['predictions']):
        pid = patient_prediction['patient']
        pset, pindex = data_loader.id_to_index_map[pid]
        pfolder = data_loader.patient_folders[pset][pindex]
        pnrslices = data_loader.compute_nr_slices(pfolder)
        if pnrslices < 6 and not _is_empty_prediction(patient_prediction):
            print "  Removing patient %d" % pid
            _clean_prediction(patient_prediction)
            flag_edited = True
            assert _is_empty_prediction(patient_prediction)
    return flag_edited
            

all_patient_models = get_all_models(PATIENT_MODEL_ORDERED_SLICES_NAMES)


for path in all_patient_models:
    print
    print "Loading %s" % os.path.basename(path) 
    metadata = np.load(path)
    flag_edited = fix_metadata(metadata)
    if flag_edited:
        print "Changed model %s" % os.path.basename(path)
        assert not fix_metadata(metadata)
        dump_prediction(metadata, path)


