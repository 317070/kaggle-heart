import numpy as np
import re
from configuration import config
from image_transform import resize_to_make_it_fit, resize_to_make_sunny_fit, resize_and_augment_sunny, \
    resize_and_augment


def uint_to_float(img):
    return img / np.float32(255.0)

def sample_augmentation_parameters():
    augmentation_params = config().augmentation_params
    res = dict()
    for key, (a, b) in augmentation_params.iteritems():
        res[key] = config().rng.uniform(a,b)
    return res


def sunny_preprocess(chunk_x, img, chunk_y, lbl):
    image = uint_to_float(img).astype(np.float32)
    chunk_x[:, :] = resize_to_make_sunny_fit(image, output_shape=chunk_x.shape[-2:])
    segmentation = lbl.astype(np.float32)
    chunk_y[:] = resize_to_make_sunny_fit(segmentation, output_shape=chunk_y.shape[-2:])


def sunny_preprocess_with_augmentation(chunk_x, img, chunk_y, lbl):

    augmentation_parameters = sample_augmentation_parameters()
    image = uint_to_float(img).astype(np.float32)
    chunk_x[:, :] = resize_and_augment_sunny(image, output_shape=chunk_x.shape[-2:], augment=augmentation_parameters)
    segmentation = lbl.astype(np.float32)
    chunk_y[:] = resize_and_augment_sunny(segmentation, output_shape=chunk_y.shape[-2:], augment=augmentation_parameters)


def sunny_preprocess_validation(chunk_x, img, chunk_y, lbl):
    image = uint_to_float(img).astype(np.float32)
    chunk_x[:, :] = resize_to_make_sunny_fit(image, output_shape=chunk_x.shape[-2:])
    segmentation = lbl.astype(np.float32)
    chunk_y[:] = resize_to_make_sunny_fit(segmentation, output_shape=chunk_y.shape[-2:])



def preprocess(patient_data, result, index):
    """
    Load the resulting data, augment it if needed, and put it in result at the correct index
    :param patient_data:
    :param result:
    :param index:
    :return:
    """
    for tag, data in patient_data.iteritems():
        desired_shape = result[tag][index].shape
        # try to fit data into the desired shape
        if tag.startswith("sliced:data"):
            # put time dimension first, then axis dimension
            patient_4d_tensor = np.swapaxes(
                                        resize_to_make_it_fit(patient_data[tag], output_shape=desired_shape[-2:])
                                        ,1,0)
            patient_shape = patient_4d_tensor.shape

            # TODO: find a better way to adapt the number of images per patient
            t_sh = [min(l1,l2) for l1, l2 in zip(desired_shape, patient_shape)]

            result[tag][index][:t_sh[0],:t_sh[1],:t_sh[2],:t_sh[3]] = patient_4d_tensor[:t_sh[0],:t_sh[1],:t_sh[2],:t_sh[3]]
        if tag.startswith("sliced:data:shape"):
            result[tag][index] = patient_data[tag]
        if tag.startswith("sliced:meta:"):
            # TODO: this probably doesn't work very well yet
            result[tag][index] = patient_data[tag]
    return


def preprocess_with_augmentation(patient_data, result, index):
    """
    Load the resulting data, augment it if needed, and put it in result at the correct index
    :param patient_data:
    :param result:
    :param index:
    :return:
    """
    augmentation_parameters = sample_augmentation_parameters()

    for tag, data in patient_data.iteritems():
        desired_shape = result[tag][index].shape
        # try to fit data into the desired shape
        if tag.startswith("sliced:data"):
            # put time dimension first, then axis dimension

            patient_4d_tensor = np.swapaxes(
                                        resize_and_augment(patient_data[tag], output_shape=desired_shape[-2:], augment=augmentation_parameters)
                                        ,1,0)
            patient_shape = patient_4d_tensor.shape

            # TODO: find a better way to adapt the number of images per patient
            t_sh = [min(l1,l2) for l1, l2 in zip(desired_shape, patient_shape)]

            result[tag][index][:t_sh[0],:t_sh[1],:t_sh[2],:t_sh[3]] = patient_4d_tensor[:t_sh[0],:t_sh[1],:t_sh[2],:t_sh[3]]
        if tag.startswith("sliced:data:shape"):
            result[tag][index] = patient_data[tag]
        if tag.startswith("sliced:meta:"):
            # TODO: this probably doesn't work very well yet
            result[tag][index] = patient_data[tag]
    return
