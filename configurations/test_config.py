import numpy as np

rng = np.random.RandomState(42)
patch_size = (128, 128)
mm_patch_size = (128, 128)
train_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': mm_patch_size,
    'rotation_range': (0, 0),
    'mask_roi': True,
    'translation_range_x': (0, 0),
    'translation_range_y': (0, 0),
    'shear_range': (0, 0),
    'roi_scale_range': (1, 1),
    'do_flip': (False, False),
    'zoom_range': (1 / 1.5, 1.5),
    'sequence_shift': False,
    'brightness_range': (0.5, 2.),
    'contrast_range': (0.7, 1.3)
}

valid_transformation_params = {
    'patch_size': patch_size,
    'mm_patch_size': mm_patch_size,
    'mask_roi': True
}
