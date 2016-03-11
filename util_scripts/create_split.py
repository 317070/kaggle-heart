import glob
import numpy as np
import re
from configuration import config
import cPickle as pickle
import utils
from validation_set import get_cross_validation_indices
import random

print "Loading data"


patient_folders = sorted(glob.glob("/data/dsb15_pkl/pkl_train/*/study/"), key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))  # glob is non-deterministic!

validation_patients_indices = get_cross_validation_indices(indices=range(1,501), validation_index=0)
train_patients_indices = [i for i in range(1,501) if i not in validation_patients_indices]

VALIDATION_REGEX = "|".join(["(/%d/)"%i for i in validation_patients_indices])

train_patient_folders = [folder for folder in patient_folders if re.search(VALIDATION_REGEX, folder) is None]
validation_patient_folders = [folder for folder in patient_folders if folder not in train_patient_folders]

import os
import os.path

def copy(from_folder, to_folder):
    command = "cp -r %s %s/."%(from_folder, to_folder)
    print command
    os.system(command)

for folder in train_patient_folders:
    f = os.path.dirname(os.path.abspath(folder))
    copy(f, "/data/dsb15_pkl/pkl_splitted/train")

for folder in validation_patient_folders:
    f = os.path.dirname(os.path.abspath(folder))
    copy(f, "/data/dsb15_pkl/pkl_splitted/valid")