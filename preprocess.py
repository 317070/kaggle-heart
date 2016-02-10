import numpy as np
import re
from configuration import config
from image_transform import resize_to_make_it_fit, resize_to_make_sunny_fit, resize_and_augment_sunny, \
    resize_and_augment, normscale_resize_and_augment
import quasi_random
from itertools import izip
from functools import partial

def uint_to_float(img):
    return img / np.float32(255.0)


quasi_random_generator = None

def sample_augmentation_parameters():
    global quasi_random_generator
    augmentation_params = config().augmentation_params
    if quasi_random_generator is None:
        quasi_random_generator = quasi_random.scrambled_halton_sequence_generator(dimension=len(config().augmentation_params),
                                                                     permutation='Braaten-Weller')
    res = dict()
    sample = quasi_random_generator.next()
    for rand, (key, (a, b)) in izip(sample, augmentation_params.iteritems()):
        #res[key] = config().rng.uniform(a,b)
        res[key] = a + rand*(b-a)
    return res


def put_in_the_middle(target_tensor, data_tensor):
    """
    put data_sensor with arbitrary number of dimensions in the middle of target tensor.
    if data_Sensor is bigger, data is cut off
    if target_sensor is bigger, original values (probably zeros) are kept
    :param target_tensor:
    :param data_tensor:
    :return:
    """
    target_shape = target_tensor.shape
    data_shape = data_tensor.shape

    def get_indices(target_width, data_width):
        if target_width>data_width:
            diff = target_width - data_width
            target_slice = slice(diff/2, target_width-(diff-diff/2))
            data_slice = slice(None, None)
        else:
            diff = data_width - target_width
            data_slice = slice(diff/2, data_width-(diff-diff/2))
            target_slice = slice(None, None)
        return target_slice, data_slice

    t_sh = [get_indices(l1,l2) for l1, l2 in zip(target_shape, data_shape)]
    target_indices, data_indices = zip(*t_sh)
    target_tensor[target_indices] = data_tensor[data_indices]


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


def preprocess_normscale(patient_data, result, index, augment=True,
                         metadata=None):
    """Normalizes scale and augments the data.
    """
    augmentation_params = sample_augmentation_parameters() if augment else None

    # Iterate over different sorts of data
    for tag, data in patient_data.iteritems():
        desired_shape = result[tag][index].shape
        

        if tag.startswith("sliced:data:singleslice"):
            print 'not rotated yet'
            data = clean_images([patient_data[tag]], metadata=metadata)
            print 'now its rotated'
            patient_4d_tensor = normscale_resize_and_augment(
                data, output_shape=desired_shape[-2:],
                augment=augmentation_params,
                pixel_spacing=metadata["PixelSpacing"])[0]

            if "area_per_pixel:sax" in result:
                raise NotImplementedError()

#            put_in_the_middle(result[tag][index], patient_4d_tensor)
            # For now, simply copy the data
            result[tag+':raw_%d' % index] = patient_4d_tensor

        elif tag.startswith("sliced:data:shape"):
            raise NotImplementedError()
        
        elif tag.startswith("sliced:data"):
            raise NotImplementedError()

        elif tag.startswith("sliced:meta:"):
            # TODO: this probably doesn't work very well yet
            result[tag][index] = patient_data[tag]


def preprocess_with_augmentation(patient_data, result, index, augment=True, metadata=None):
    """
    Load the resulting data, augment it if needed, and put it in result at the correct index
    :param patient_data:
    :param result:
    :param index:
    :return:
    """
    if augment:
        augmentation_parameters = sample_augmentation_parameters()
    else:
        augmentation_parameters = None

    for tag, data in patient_data.iteritems():
        desired_shape = result[tag][index].shape
        # try to fit data into the desired shape
        if tag.startswith("sliced:data:singleslice"):
            data = clean_images([patient_data[tag]], metadata=metadata)
            patient_4d_tensor, zoom_ratios = resize_and_augment(data, output_shape=desired_shape[-2:], augment=augmentation_parameters)[0]
            if "area_per_pixel:sax" in result:
                result["area_per_pixel:sax"][index] = zoom_ratios[0] * np.prod(metadata["PixelSpacing"])

            put_in_the_middle(result[tag][index], patient_4d_tensor)
        elif tag.startswith("sliced:data"):
            # put time dimension first, then axis dimension
            data = clean_images(patient_data[tag], metadata=metadata)
            patient_4d_tensor, zoom_ratios = resize_and_augment(data, output_shape=desired_shape[-2:], augment=augmentation_parameters)
            if "area_per_pixel:sax" in result:
                result["area_per_pixel:sax"][index] = zoom_ratios[0] * np.prod(metadata["PixelSpacing"])

            if "noswitch" not in tag:
                patient_4d_tensor = np.swapaxes(patient_4d_tensor,1,0)

            put_in_the_middle(result[tag][index], patient_4d_tensor)
        if tag.startswith("sliced:data:shape"):
            result[tag][index] = patient_data[tag]
        if tag.startswith("sliced:meta:"):
            # TODO: this probably doesn't work very well yet
            result[tag][index] = patient_data[tag]
    return

preprocess = partial(preprocess_with_augmentation, augment=False)


def clean_images(data, metadata):
    """
    clean up 4d-tensor of imdata consistently (fix contrast, move upside up, etc...)
    :param data:
    :return:
    """
    for process in config().cleaning_processes:
        data = process(data, metadata)
    return data


def normalize_contrast(imdata, metadata=None):
    # normalize contrast
    flat_data = np.concatenate([i.flatten() for i in imdata]).flatten()
    high = np.percentile(flat_data, 95.0)
    low  = np.percentile(flat_data, 5.0)
    for i in xrange(len(imdata)):
        image = imdata[i]
        image = 1.0 * (image - low) / (high - low)
        image = np.clip(image, 0.0, 1.0)
        imdata[i] = image

    return imdata


def set_upside_up(data, metadata=None):
    print 'rotating'
    out_data = []
    for idx, dslice in enumerate(data):
        out_data.append(set_upside_up_slice(dslice, metadata))
    return out_data


def set_upside_up_slice(dslice, metadata=None):
    # turn upside up
    print metadata
    F = np.array(metadata["ImageOrientationPatient"]).reshape((2, 3))

    f_1 = F[1, :] / np.linalg.norm(F[1, :])
    f_2 = F[0, :] / np.linalg.norm(F[0, :])

    x_e = np.array([1, 0, 0])
    y_e = np.array([0, 1, 0])

    if abs(np.dot(y_e, f_1)) >= abs(np.dot(y_e, f_2)):
        out_data = np.transpose(dslice, (0, 2, 1))
        f_1, f_2 = f_2, f_1
    else:
        out_data = dslice

    if np.dot(y_e, f_1) < 0:
        out_data = out_data[:, ::-1, :]

    if np.dot(x_e, f_2) < 0:
        out_data = out_data[:, :, ::-1]

    return out_data