"""Visualise errors based on the per slice predictions.
"""

import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import data_loader


_extract_id_from_path = lambda path: int(re.search(r'/(\d+)/', path).group(1))

patient_ids = {
    set: [
        _extract_id_from_path(folder) 
        for folder in data_loader.patient_folders[set]]
    for set in data_loader.patient_folders}


def compute_average_crps(patient_preds):
    crpss = [slicepreds['crps'] for slicepreds in patient_preds['slices'].values()]


def visu_preds(preds):
    preds_per_slice = preds['predictions_per_slice']
    valid_ids = patient_ids['validation']
    valid_preds = [preds_per_slice[id] for id in valid_ids]
    for i in valid_preds[0]['slices']:
        print valid_preds[0]['slices'][i].keys()
#    valid_preds.sort(key=lambda x: x[])


if __name__ == '__main__':
    preds_filename = sys.argv[1]
    preds = np.load(preds_filename)
    visu_preds(preds)