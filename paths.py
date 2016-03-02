import json

with open('SETTINGS.json') as data_file:
    paths = json.load(data_file)

print paths

TRAIN_DATA_PATH = paths["TRAIN_DATA_PATH"]
MODEL_PATH = paths["MODEL_PATH"]
VALIDATE_DATA_PATH = paths["VALIDATE_DATA_PATH"]
PKL_TRAIN_DATA_PATH = paths["PKL_TRAIN_DATA_PATH"]
PKL_VALIDATE_DATA_PATH = paths["PKL_VALIDATE_DATA_PATH"]
