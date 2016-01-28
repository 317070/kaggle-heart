import numpy as np

import data_iterators

rng = np.random.RandomState(42)
patch_size = (64, 64)
train_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': (-16, 16),
    'translation_range': (-8, 8),
    'shear_range': (0,0)
}

valid_transformation_params = {
    'patch_size': (64, 64),
    'rotation_range': None,
    'translation_range': None,
    'shear_range': None
}

# TODO
batch_size = 128
num_batches_train = 500
learning_rate_schedule = {}
validate_every = 20
save_every = 20