"""Library implementing the data augmentations.
"""
import numpy as np
import skimage.io
import skimage.transform

from custom_warnings import deprecated


tform_identity = skimage.transform.AffineTransform()
NO_AUGMENT_PARAMS = {
    "zoom_x": 1.0,
    "zoom_y": 1.0,
    "rotate": 0.0,
    "shear": 0.0,
    "skew_x": 0.0,
    "skew_y": 0.0,
    "translate_x": 0.0,
    "translate_y": 0.0,
    "flip_vert": 0.0,
    "roll_time": 0.0,
    "flip_time": 0.0,
    "change_brightness": 0.0,
}

def resize_to_make_it_fit(images, output_shape=(50, 50)):
    """Resizes the images to a given shape.
    """
    max_time = max(images[i].shape[0] for i in xrange(len(images)))
    final_shape = (len(images),max_time) + output_shape
    result = np.zeros(final_shape, dtype="float32")

    volume_change = []
    #result.reshape((final_shape[0],-1) + output_shape)
    for i, mri_slice in enumerate(images):
        mri_slice = mri_slice.reshape((-1,)+mri_slice.shape[-2:])
        scaling = max(mri_slice[0].shape[-2]/output_shape[-2], mri_slice[0].shape[-1]/output_shape[-1])
        tform = build_rescale_transform(scaling, mri_slice[0].shape[-2:], target_shape=output_shape)

        for j, frame in enumerate(mri_slice):
            # TODO: can't this be done better?
            result[i,j] = fast_warp(frame, tform, output_shape=output_shape)

        A = tform.params[:2, :2]
        volume_change.append(np.linalg.norm(A[:,0]) * np.linalg.norm(A[:,1]))
        assert tform.params[2,2] == 1, (tform.params[2,2],)

    #result.reshape(final_shape)
    return result, volume_change


@deprecated
def normscale_resize_and_augment(slices, output_shape=(50, 50), augment=None,
                                 pixel_spacing=(1,1), shift_center=(.4, .5),
                                 normalised_patch_size=(200,200)):
    """Normalizes the scale, augments, and crops the image.

    WARNING: This function contains bugs. We kept it around to ensure older
    models would still behave in the same way. Use normscale_resize_and_augment_2
    instead.
    """
    if not pixel_spacing[0] == pixel_spacing[1]:
        raise NotImplementedError("Only supports square pixels")

    # No augmentation:
    if augment is None:
        augment = NO_AUGMENT_PARAMS

    current_shape = slices[0].shape[-2:]
    normalised_shape = tuple(int(float(d)*ps) for d,ps in zip(current_shape, pixel_spacing))

    max_time = max(slices[i].shape[0] for i in xrange(len(slices)))
    final_shape = (len(slices),max_time) + output_shape
    result = np.zeros(final_shape, dtype="float32")

    for i, mri_slice in enumerate(slices):
        # For each slice, build a transformation that extracts the right patch,
        # and augments the data.
        # First, we scale the images such that they all have the same scale
        norm_rescaling = 1./pixel_spacing[0]
        tform_normscale = build_rescale_transform(
            norm_rescaling, mri_slice[0].shape[-2:], target_shape=normalised_shape)
        # Next, we shift the center of the image to the left (assumes upside_up normalisation)
        tform_shift_center, tform_shift_uncenter = (
            build_shift_center_transform(
                normalised_shape, shift_center, normalised_patch_size))

        # zooming is OK
        augment_tform = build_augmentation_transform(**augment)

        patch_scale = max(
            normalised_patch_size[0]/output_shape[0],
            normalised_patch_size[1]/output_shape[1])
        tform_patch_scale = build_rescale_transform(
            patch_scale, normalised_patch_size, target_shape=output_shape)

        # x and y axis transform
        total_tform = tform_patch_scale + tform_shift_uncenter + augment_tform + tform_shift_center + tform_normscale

        # Time axis transform
        t_map = range(mri_slice.shape[0])
        if "roll_time" in augment:
            t_map = np.roll(t_map, int(np.floor(augment["roll_time"])))
        if "flip_time" in augment and augment["flip_time"] > 0.5:
            t_map = t_map[::-1]

        for j, frame in enumerate(mri_slice):
            j_shifted = t_map[j]
            result[i,j_shifted] = fast_warp(frame, total_tform, output_shape=output_shape)

    return result


NRMSC_DEFAULT_SHIFT_CENTER = (.4, .5)
def normscale_resize_and_augment_2(slices, output_shape=(50, 50), augment=None,
                                   pixel_spacing=(1,1), shift_center=(None, None),
                                   normalised_patch_size=(200,200)):
    """Normalizes the scale, augments, and crops the image.
    """
    if not pixel_spacing[0] == pixel_spacing[1]:
        raise NotImplementedError("Only supports square pixels")

    if shift_center == (None, None):
        shift_center = NRMSC_DEFAULT_SHIFT_CENTER

    # No augmentation:
    if augment is None:
        augment = NO_AUGMENT_PARAMS

    current_shape = slices[0].shape[-2:]
    normalised_shape = tuple(int(float(d)*ps) for d,ps in zip(current_shape, pixel_spacing))

    max_time = max(slices[i].shape[0] for i in xrange(len(slices)))
    final_shape = (len(slices),max_time) + output_shape
    result = np.zeros(final_shape, dtype="float32")

    for i, mri_slice in enumerate(slices):
        # For each slice, build a transformation that extracts the right patch,
        # and augments the data.
        # First, we scale the images such that they all have the same scale
        norm_rescaling = 1./pixel_spacing[0]
        tform_normscale = build_rescale_transform(
            norm_rescaling, mri_slice[0].shape[-2:], target_shape=normalised_shape)
        # Next, we shift the center of the image to the left (assumes upside_up normalisation)
        tform_shift_center, tform_shift_uncenter = (
            build_shift_center_transform(
                normalised_shape, shift_center, normalised_patch_size))

        augment_tform = build_augmentation_transform(**augment)

        patch_scale = max(
            float(normalised_patch_size[0])/output_shape[0],
            float(normalised_patch_size[1])/output_shape[1])

        tform_patch_scale = build_rescale_transform(
            patch_scale, normalised_patch_size, target_shape=output_shape)

        # x and y axis transform
        total_tform = tform_patch_scale + tform_shift_uncenter + augment_tform + tform_shift_center + tform_normscale

        # Time axis transform
        t_map = range(mri_slice.shape[0])
        if "roll_time" in augment:
            t_map = np.roll(t_map, int(np.floor(augment["roll_time"])))
        if "flip_time" in augment and augment["flip_time"] > 0.5:
            t_map = t_map[::-1]

        for j, frame in enumerate(mri_slice):
            j_shifted = t_map[j]
            result[i,j_shifted] = fast_warp(frame, total_tform, output_shape=output_shape)

    return result


def resize_and_augment(images, output_shape=(50, 50), augment=None):
    if augment is None:
        return resize_to_make_it_fit(images, output_shape=output_shape)

    max_time = max(images[i].shape[0] for i in xrange(len(images)))
    final_shape = (len(images),max_time) + output_shape
    result = np.zeros(final_shape, dtype="float32")

    volume_change = []
    #result.reshape((final_shape[0],-1) + output_shape)
    for i, mri_slice in enumerate(images):
        mri_slice = mri_slice.reshape((-1,)+mri_slice.shape[-2:])
        scaling = max(1.0*mri_slice[0].shape[-2]/output_shape[-2], 1.0*mri_slice[0].shape[-1]/output_shape[-1])
        tform = build_rescale_transform(scaling, mri_slice[0].shape[-2:], target_shape=output_shape)
        # add rotation
        # add skew
        # add translation
        tform_center, tform_uncenter = build_center_uncenter_transforms(mri_slice[0].shape[-2:])
        augment_tform = build_augmentation_transform((1.0, 1.0), augment["rotation"], augment["shear"], augment["translation"], flip=False)
        total_tform = tform + tform_uncenter + augment_tform + tform_center
        for j, frame in enumerate(mri_slice):
            result[i,j] = fast_warp(frame, total_tform, output_shape=output_shape)

        A = total_tform.params[:2, :2]
        volume_change.append(np.linalg.norm(A[:,0]) * np.linalg.norm(A[:,1]))
        assert total_tform.params[2,2] == 1, (total_tform.params[2,2],)

    #result.reshape(final_shape)
    return result, volume_change


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


def build_shift_center_transform(image_shape, center_location, patch_size):
    """Shifts the center of the image to a given location.

    This function tries to include as much as possible of the image in the patch
    centered around the new center. If the patch arount the ideal center
    location doesn't fit within the image, we shift the center to the right so
    that it does.
    """
    center_absolute_location = [
        center_location[0]*image_shape[1], center_location[1]*image_shape[0]]

    # Check for overlap at the edges
    center_absolute_location[0] = max(
        center_absolute_location[0], patch_size[1]/2.0)
    center_absolute_location[1] = max(
        center_absolute_location[1], patch_size[0]/2.0)
    center_absolute_location[0] = min(
        center_absolute_location[0], image_shape[1] - patch_size[1]/2.0)
    center_absolute_location[1] = min(
        center_absolute_location[1], image_shape[0] - patch_size[0]/2.0)

    # Check for overlap at both edges
    if patch_size[0] > image_shape[0]:
        center_absolute_location[1] = image_shape[0] / 2.0
    if patch_size[1] > image_shape[1]:
        center_absolute_location[0] = image_shape[1] / 2.0

    # Build transform
    new_center = np.array(center_absolute_location)
    translation_center = new_center - 0.5
    translation_uncenter = -np.array((patch_size[1]/2.0, patch_size[0]/2.0)) - 0.5
    return (
        skimage.transform.SimilarityTransform(translation=translation_center),
        skimage.transform.SimilarityTransform(translation=translation_uncenter))


def build_augmentation_transform(zoom_x=1.0,
                                 zoom_y=1.0,
                                 skew_x=0,
                                 skew_y=0,
                                 rotate=0,
                                 shear=0,
                                 translate_x=0,
                                 translate_y=0,
                                 flip=False,
                                 flip_vert=False,
                                 **kwargs):

    #print "Not performed transformations:", kwargs.keys()

    if flip > 0.5:
        shear += 180
        rotate += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    if flip_vert > 0.5:
        shear += 180

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom_x, 1/zoom_y), rotation=np.deg2rad(rotate), shear=np.deg2rad(shear), translation=(translate_x, translate_y))
    skew_x = np.deg2rad(skew_x)
    skew_y = np.deg2rad(skew_y)
    tform_skew = skimage.transform.ProjectiveTransform(matrix=np.array([[np.tan(skew_x)*np.tan(skew_y) + 1, np.tan(skew_x), 0],
                                                                        [np.tan(skew_y), 1, 0],
                                                                        [0, 0, 1]]))
    return tform_skew + tform_augment

@deprecated
def random_perturbation_transform(zoom_range=[1.0, 1.0], rotation_range=[0.0, 0.0], skew_x_range=[0.0, 0.0], skew_y_range=[0.0, 0.0], shear_range=[0.0, 0.0], translation_range=[0.0, 0.0], do_flip=True, allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translate = (shift_x, shift_y)

    rotate = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    skew_x = rng.uniform(*skew_x_range)
    skew_y = rng.uniform(*skew_y_range)

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

    return build_augmentation_transform(zoom_x=zoom_x,
                                        zoom_y=zoom_y,
                                        skew_x=skew_x,
                                        skew_y=skew_y,
                                        rotate=rotate,
                                        shear=shear,
                                        translate_x=translate[0],
                                        translate_y=translate[1],
                                        flip=flip
                                        )

@deprecated
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
@deprecated
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
@deprecated
def perturb_rescaled_fixed(img, scale, tform_augment, target_shape=(50, 50)):
    """
    scale is a DOWNSCALING factor.
    """
    tform_rescale = build_rescale_transform(scale, img.shape, target_shape) # also does centering

    tform_center, tform_uncenter = build_center_uncenter_transforms(img.shape)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_rescale + tform_augment, output_shape=target_shape, mode='constant').astype('float32')

