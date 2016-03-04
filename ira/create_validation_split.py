import numpy as np
import glob
import re
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


def save_train_validation_ids(filename, data_path):
    patient_dirs = sorted(glob.glob(data_path + "/*/study/"),
                          key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))
    dirs_indices = range(0, len(patient_dirs))

    valid_dirs_indices = get_cross_validation_indices(indices=dirs_indices, validation_index=0)
    train_patient_indices = list(set(dirs_indices) - set(valid_dirs_indices))

    train_patient_dirs = [utils.get_patient_id(patient_dirs[idx]) for idx in train_patient_indices]
    validation_patient_dirs = [utils.get_patient_id(patient_dirs[idx]) for idx in valid_dirs_indices]

    d = {'train': train_patient_dirs, 'valid': validation_patient_dirs}
    utils.save_pkl(d, filename)
    print 'train-valid patients split saved to', filename
    return d


if __name__ == '__main__':
    global_data_path = '/data/dsb15_pkl/pkl_train'

    p = save_train_validation_ids(global_data_path)
    print 'TRAIN'
    for path in p['train']:
        print utils.get_patient_id(path),

    print '\nVALID'
    valid_ids = []
    for path in p['valid']:
        valid_ids.append(utils.get_patient_id(path))
        print utils.get_patient_id(path),

    valid_ids1 = []
    g = glob.glob('/data/dsb15_pkl/pkl_splitted/valid/*/study/')
    for path in g:
        valid_ids1.append(utils.get_patient_id(path))
    print set(valid_ids) == set(valid_ids1)
