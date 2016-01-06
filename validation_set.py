import glob
import utils
import numpy as np
import re
from configuration import config
import cPickle as pickle

def get_cross_validation_indices(validation_index=0, number_of_cross_validations = 6):
    patient_folders = glob.glob("/data/dsb15_pkl/4d_group_pkl_train/*/study/")
    np.random.seed(317070)  # because I can

    indices = range(1,501)
      # 16.6 - 83.3
    patients_per_cross_validation = len(indices) // number_of_cross_validations

    cross_validations = []

    for i in xrange(number_of_cross_validations):
        if len(indices)>patients_per_cross_validation:
            validation_patients_indices = list(np.random.choice(indices, patients_per_cross_validation, replace=False))
        else:
            validation_patients_indices = indices
        indices = [index for index in indices if index not in validation_patients_indices]
        cross_validations.append(validation_patients_indices)

    return cross_validations[validation_index]

