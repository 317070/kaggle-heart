"""
take in a numpy tensor, and reshape last 2 dimensions as images to fit the desired shape
"""
import glob
import os
import multiprocessing as mp
import numpy as np
import skimage.io
import skimage.transform
import utils



tform_identity = skimage.transform.AffineTransform()

def resize_to_make_it_fit(images, output_shape=(50, 50)):
    final_shape = (len(images),) + images[0].shape[:-2] + output_shape
    result = np.zeros(final_shape, dtype="float32")

    #result.reshape((final_shape[0],-1) + output_shape)
    for i, mri_slice in enumerate(images):
        mri_slice = mri_slice.reshape((-1,)+mri_slice.shape[-2:])
        scaling = max(mri_slice[0].shape[-2]/output_shape[-2], mri_slice[0].shape[-1]/output_shape[-1])
        tform = build_rescale_transform(scaling, mri_slice[0].shape[-2:], target_shape=output_shape)

        for j, frame in enumerate(mri_slice):
            # TODO: can't this be done better?
            result[i,j] = fast_warp(frame, tform, output_shape=output_shape)

    #result.reshape(final_shape)
    return result


def resize_and_augment(images, output_shape=(50, 50), augment=None):
    final_shape = (len(images),) + images[0].shape[:-2] + output_shape
    result = np.zeros(final_shape, dtype="float32")

    #result.reshape((final_shape[0],-1) + output_shape)
    for i, mri_slice in enumerate(images):
        mri_slice = mri_slice.reshape((-1,)+mri_slice.shape[-2:])
        scaling = max(mri_slice[0].shape[-2]/output_shape[-2], mri_slice[0].shape[-1]/output_shape[-1])
        tform = build_rescale_transform(scaling, mri_slice[0].shape[-2:], target_shape=output_shape)
        # add rotation
        # add skew
        # add translation
        tform_center, tform_uncenter = build_center_uncenter_transforms(mri_slice[0].shape[-2:])
        augment_tform = build_augmentation_transform((1.0, 1.0), augment["rotation"], augment["shear"], augment["translation"], flip=False)
        total_tform = tform + tform_uncenter + augment_tform + tform_center
        for j, frame in enumerate(mri_slice):
            result[i,j] = fast_warp(frame, total_tform, output_shape=output_shape)

    #result.reshape(final_shape)
    return result


def resize_to_make_sunny_fit(image, output_shape=(50, 50)):
    scaling = max(image.shape[-2]/output_shape[-2], image.shape[-1]/output_shape[-1])
    tform = build_rescale_transform(scaling, image.shape[-2:], target_shape=output_shape)
    return fast_warp(image, tform, output_shape=output_shape)


def resize_and_augment_sunny(image, output_shape=(50, 50), augment=None):
    if augment is None:
        return resize_to_make_sunny_fit(image, output_shape=(50, 50))

    final_shape = image.shape[:-2] + output_shape
    result = np.zeros(final_shape, dtype="float32")

    #result.reshape((final_shape[0],-1) + output_shape)
    scaling = max(image.shape[-2]/output_shape[-2], image.shape[-1]/output_shape[-1])
    tform = build_rescale_transform(scaling, image.shape[-2:], target_shape=output_shape)
    # add rotation
    # add skew
    # add translation
    tform_center, tform_uncenter = build_center_uncenter_transforms(image.shape[-2:])
    augment_tform = build_augmentation_transform((1.0, 1.0), augment["rotation"], augment["shear"], augment["translation"], flip=False)
    total_tform = tform + tform_uncenter + augment_tform + tform_center
    #result.reshape(final_shape)
    return fast_warp(image, total_tform, output_shape=output_shape, mode='constant')


def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params # tf._matrix is
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)


def build_centering_transform(image_shape, target_shape=(50, 50)):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_rescale_transform(downscale_factor, image_shape, target_shape):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(scale=(downscale_factor, downscale_factor))

    # centering
    shift_x = cols / (2.0 * downscale_factor) - tcols / 2.0
    shift_y = rows / (2.0 * downscale_factor) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True, allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0) # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch_x = np.exp(rng.uniform(*log_stretch_range))
        stretch_y = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch_x
        zoom_y = zoom * stretch_y
    elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)


def perturb(img, augmentation_params, target_shape=(50, 50), rng=np.random):
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    tform_centering = build_centering_transform(img.shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape)
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_centering + tform_augment, output_shape=target_shape, mode='constant').astype('float32')

## RESCALING

def perturb_rescaled(img, scale, augmentation_params, target_shape=(50, 50), rng=np.random):
    """
    scale is a DOWNSCALING factor.
    """
    tform_rescale = build_rescale_transform(scale, img.shape, target_shape) # also does centering
    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape)
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_rescale + tform_augment, output_shape=target_shape, mode='constant').astype('float32')

# for test-time augmentation
def perturb_rescaled_fixed(img, scale, tform_augment, target_shape=(50, 50)):
    """
    scale is a DOWNSCALING factor.
    """
    tform_rescale = build_rescale_transform(scale, img.shape, target_shape) # also does centering

    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_rescale + tform_augment, output_shape=target_shape, mode='constant').astype('float32')

