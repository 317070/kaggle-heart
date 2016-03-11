"""Module responsible for generating the train-validation split.
"""
import cPickle as pickle
import glob
import re

import numpy as np

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
        List of validation set indices belonging to the requested split.
    """
    np.random.seed(rng_seed)

      # 16.6 - 83.3
    samples_per_split = len(indices) // number_of_splits

    cross_validations = []

    for _ in xrange(number_of_splits):
        if len(indices)>samples_per_split:
            validation_patients_indices = list(np.random.choice(indices, samples_per_split, replace=False))
        else:
            validation_patients_indices = indices
        indices = [index for index in indices if index not in validation_patients_indices]
        cross_validations.append(validation_patients_indices)

    return cross_validations[validation_index]