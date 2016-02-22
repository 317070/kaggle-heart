import numpy as np
rng = np.random.RandomState(42)
patch_size = (64, 64)
mm_patch_size = (128, 128)
train_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': mm_patch_size,
    'rotation_range': (-180, 180),
    'mask_roi': False,
    'translation_range_x': (-5, 10),
    'translation_range_y': (-10, 5),
    'shear_range': (0, 0),
    'roi_scale_range': (0.8, 1.2),
    'do_flip': True,
    'sequence_shift': False
}

valid_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': mm_patch_size,
    'mask_roi': False
}
