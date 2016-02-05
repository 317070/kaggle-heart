import cPickle as pickle
import glob
import re
import numpy as np
import skimage.io
import skimage.transform
from configuration import config
import timeit


def read_labels(file_path):
    id2labels = {}
    train_csv = open(file_path)
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, systole, diastole = item.replace('\n', '').split(',')
        id2labels[int(id)] = [float(systole), float(diastole)]
    return id2labels


def read_patients_data(paths):
    data_dict = {}
    for p in paths:
        patient_id = int(re.search(r'/(\d+)/', p).group(1))
        data_dict[patient_id] = read_patient(p)
    return data_dict


def read_patient(path):
    slices_paths = sorted(glob.glob(path + '/*.pkl'))
    slices_dict = {}
    for sp in slices_paths:
        slice = re.search(r'/(\w+\d+)*\.pkl$', sp).group(1)
        slices_dict[slice] = read_slice(sp)
    return slices_dict


def read_slice(path):
    d = pickle.load(open(path))['data']
    return d


def read_slice_with_metadata(path):
    d = pickle.load(open(path))
    img_data = d['data']
    m_data = d['metadata']
    # TODO select relevant fields from metadata
    return img_data, m_data


def sample_augmentation_parameters(transformation):
    random_params = None
    if all([transformation['rotation_range'],
            transformation['translation_range'],
            transformation['shear_range']]):
        shift_x = config().rng.uniform(*transformation['translation_range'])
        shift_y = config().rng.uniform(*transformation['translation_range'])
        translation = (shift_x, shift_y)
        rotation = config().rng.uniform(*transformation['rotation_range'])
        shear = config().rng.uniform(*transformation['shear_range'])
        random_params = {'translation': translation, 'rotation': rotation, 'shear': shear}
    return random_params


def transform(data, transformation, random_augmentation_params=None):
    """
    :param data: (30, height, width) matrix from one slice of MRI
    :param transformation:
    :return:
    """
    out_shape = (30,) + transformation['patch_size']
    out_data = np.zeros(out_shape, dtype='float32')

    # if no random params are given -> sample new params
    # if there is no augmentations random_augmentation_params = None
    if not random_augmentation_params:
        random_augmentation_params = sample_augmentation_parameters(transformation)

    for i in xrange(data.shape[0]):
        scaling = max(1. * data.shape[-2] / out_shape[-2], 1. * data.shape[-1] / out_shape[-1])

        tform = build_rescale_transform(scaling, data.shape[-2:], target_shape=transformation['patch_size'])
        total_tform = tform

        if random_augmentation_params:
            tform_center, tform_uncenter = build_center_uncenter_transforms(data.shape[-2:])

            augment_tform = build_augmentation_transform(rotation=random_augmentation_params['rotation'],
                                                         shear=random_augmentation_params['shear'],
                                                         translation=random_augmentation_params['translation'])
            total_tform = tform + tform_uncenter + augment_tform + tform_center

        out_data[i] = fast_warp(data[i], total_tform, output_shape=transformation['patch_size'])

    # if the sequence is < 30 timesteps, copy images from the beginning of the sequence
    if data.shape[0] < out_shape[0]:
        for i, j in enumerate(xrange(data.shape[0], out_shape[0])):
            out_data[j] = out_data[i]
    # if > 30, remove images
    if data.shape[0] > out_shape[0]:
        out_data = out_data[:30]
    return out_data


tform_identity = skimage.transform.AffineTransform()


def fast_warp(img, tf, output_shape, mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params  # tf._matrix is
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
    center_shift = np.array(
        [image_shape[1], image_shape[0]]) / 2.0 - 0.5  # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(rotation=0, shear=0, translation=(0, 0), flip=False, zoom=(1.0, 1.0)):
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1 / zoom[0], 1 / zoom[1]), rotation=np.deg2rad(rotation),
                                                      shear=np.deg2rad(shear), translation=translation)
    return tform_augment
