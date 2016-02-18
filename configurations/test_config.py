import numpy as np
rng = np.random.RandomState(42)
patch_size = (128, 128)
train_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': (-90, 90),
    'translation_range': (-10, 10),
    'shear_range': (0, 0),
    'roi_scale_range': (0.9, 1.3),
    'do_flip': False,
    'sequence_shift': False
}

valid_transformation_params = {
    'patch_size': patch_size
}
