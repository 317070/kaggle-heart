# NOT TESTED!!!!!!!!!!!!!


import numpy as np
import glob
import re
import os
import utils

_DEFAULT_SEED = 317070


def get_cross_validation_indices(indices, validation_index=0,
                                 number_of_splits=6, rng_seed=_DEFAULT_SEED):
    """Splits the indices randomly into a given number of folds.
    The data is randomly split into the chosen number of folds. The indices
    belonging to the requested fold are returned.
    IMPORTANT: As a side effect, this function changes the numpy RNG seed.
    Args:
        indices: The list of indices to be split.
        validation_index: Index of the split to return. (default = 0)
        number_of_splits: Number of splits to make. (default = 6)
        rng_seed: RNG seed to use.
    Returns:
        List of indices belonging to the requested split.
    """
    np.random.seed(rng_seed)
    samples_per_split = len(indices) // number_of_splits

    cross_validations = []

    for _ in xrange(number_of_splits):
        if len(indices) > samples_per_split:
            validation_patients_indices = list(np.random.choice(indices, samples_per_split, replace=False))
        else:
            validation_patients_indices = indices
        indices = [index for index in indices if index not in validation_patients_indices]
        cross_validations.append(validation_patients_indices)

    return cross_validations[validation_index]


def split_train_validation(global_data_path, train_data_path, valid_data_path, number_of_splits):
    print "Loading data"

    patient_dirs = sorted(glob.glob(global_data_path + "/*/study/"),
                          key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))
    dirs_indices = range(0, len(patient_dirs))

    valid_dirs_indices = get_cross_validation_indices(indices=dirs_indices, validation_index=0,
                                                      number_of_splits=number_of_splits)
    train_patient_indices = list(set(dirs_indices) - set(valid_dirs_indices))

    train_patient_dirs = [patient_dirs[idx] for idx in train_patient_indices]
    validation_patient_dirs = [patient_dirs[idx] for idx in valid_dirs_indices]

    for folder in train_patient_dirs:
        f = os.path.dirname(os.path.abspath(folder))
        utils.copy(f, train_data_path)

    for folder in validation_patient_dirs:
        f = os.path.dirname(os.path.abspath(folder))
        utils.copy(f, valid_data_path)


if __name__ == '__main__':
    # global_data_path = '/data/dsb15_pkl/pkl_train'
    # train_data_path = '/data/dsb15_pkl/pkl_splitted/train'
    # valid_data_path = '/data/dsb15_pkl/pkl_splitted/valid'
    global_data_path = '/data/dsb15_pkl/pkl_splitted/valid'
    train_data_path = '/data/dsb15_pkl/pkl_splitted/valid1'
    valid_data_path = '/data/dsb15_pkl/pkl_splitted/valid2'
    if not os.path.isdir(train_data_path):
        os.makedirs(train_data_path)
    if not os.path.isdir(valid_data_path):
        os.makedirs(valid_data_path)

    split_train_validation(global_data_path, train_data_path, valid_data_path, number_of_splits=2)
