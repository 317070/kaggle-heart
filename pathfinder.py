import json
import utils
import os

with open('SETTINGS.json') as data_file:
    paths = json.load(data_file)

MODEL_PATH = paths["MODEL_PATH"]

TRAIN_DATA_PATH = paths["TRAIN_DATA_PATH"]
PKL_TRAIN_DATA_PATH = paths["PKL_TRAIN_DATA_PATH"]
utils.check_data_paths(TRAIN_DATA_PATH, PKL_TRAIN_DATA_PATH)


VALIDATE_DATA_PATH = paths["VALIDATE_DATA_PATH"]
PKL_VALIDATE_DATA_PATH = paths["PKL_VALIDATE_DATA_PATH"]
utils.check_data_paths(VALIDATE_DATA_PATH, PKL_VALIDATE_DATA_PATH)

TRAIN_LABELS_PATH = paths["TRAIN_LABELS_PATH"]
if not os.path.isfile(TRAIN_LABELS_PATH):
        raise ValueError('no file with train labels')

