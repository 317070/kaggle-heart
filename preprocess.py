import numpy as np
import re
from configuration import config
from image_transform import resize_to_make_it_fit

def uint_to_float(img):
    return img / np.float32(255.0)

def sunny_preprocess(chunk_x, img, chunk_y, lbl):
    chunk_x[:, :] = uint_to_float(img).astype(np.float32)
    chunk_y[:] = lbl.astype(np.float32)

def sunny_preprocess_validation(chunk_x, img, chunk_y, lbl):
    chunk_x[:, :] = uint_to_float(img).astype(np.float32)
    chunk_y[:] = lbl.astype(np.float32)

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
            t_sh = patient_4d_tensor.shape
            result[tag][index][:t_sh[0],:t_sh[1],:t_sh[2],:t_sh[3]] = patient_4d_tensor
        if tag.startswith("sliced:data:shape"):
            result[tag][index] = patient_data[tag]
        if tag.startswith("sliced:meta:"):
            # TODO: this probably doesn't work very well yet
            result[tag][index] = patient_data[tag]
    return