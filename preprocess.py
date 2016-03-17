"""Library implementing different ways to preprocess the data.
"""

import re

import numpy as np
import skimage.io
import skimage.transform

from itertools import izip
from functools import partial

import quasi_random
import utils

from configuration import config
from image_transform import resize_to_make_it_fit, resize_to_make_sunny_fit, resize_and_augment_sunny, \
    resize_and_augment, normscale_resize_and_augment, build_rescale_transform, build_shift_center_transform, \
    build_augmentation_transform, build_center_uncenter_transforms, fast_warp


def uint_to_float(img):
    return img / np.float32(255.0)

DEFAULT_AUGMENTATION_PARAMETERS = {
    "zoom_x":[1, 1],
    "zoom_y":[1, 1],
    "rotate":[0, 0],
    "shear":[0, 0],
    "skew_x":[0, 0],
    "skew_y":[0, 0],
    "translate_x":[0, 0],
    "translate_y":[0, 0],
    "flip_vert": [0, 0],
    "roll_time": [0, 0],
    "flip_time": [0, 0],
    "change_brightness": [0, 0],
}

quasi_random_generator = None

def sample_augmentation_parameters():
    global quasi_random_generator

    augm = config().augmentation_params
    if "translation" in augm:
        newdict = dict()
        if "translation" in augm:
            newdict["translate_x"] = augm["translation"]
            newdict["translate_y"] = augm["translation"]
        if "shear" in augm:
            newdict["shear"] = augm["shear"]
        if "flip_vert" in augm:
            newdict["flip_vert"] = augm["flip_vert"]
        if "roll_time" in augm:
            newdict["roll_time"] = augm["roll_time"]
        if "flip_time" in augm:
            newdict["flip_time"] = augm["flip_time"]
        augmentation_params = dict(DEFAULT_AUGMENTATION_PARAMETERS, **newdict)
    else:
        augmentation_params = dict(DEFAULT_AUGMENTATION_PARAMETERS, **augm)

    if quasi_random_generator is None:
        quasi_random_generator = quasi_random.scrambled_halton_sequence_generator(dimension=len(augmentation_params),
                                                                                  permutation='Braaten-Weller')
    res = dict()
    try:
        sample = quasi_random_generator.next()
    except ValueError:
        quasi_random_generator = quasi_random.scrambled_halton_sequence_generator(dimension=len(augmentation_params),
                                                                                  permutation='Braaten-Weller')
        sample = quasi_random_generator.next()

    for rand, (key, (a, b)) in izip(sample, augmentation_params.iteritems()):
        #res[key] = config().rng.uniform(a,b)
        res[key] = a + rand*(b-a)
    return res


def sample_test_augmentation_parameters():
    global quasi_random_generator

    augm = config().augmentation_params_test if hasattr(config(), 'augmentation_params_test') else config().augmentation_params
    if "translation" in augm:
        newdict = dict()
        if "translation" in augm:
            newdict["translate_x"] = augm["translation"]
            newdict["translate_y"] = augm["translation"]
        if "shear" in augm:
            newdict["shear"] = augm["shear"]
        if "flip_vert" in augm:
            newdict["flip_vert"] = augm["flip_vert"]
        if "roll_time" in augm:
            newdict["roll_time"] = augm["roll_time"]
        if "flip_time" in augm:
            newdict["flip_time"] = augm["flip_time"]
        augmentation_params = dict(DEFAULT_AUGMENTATION_PARAMETERS, **newdict)
    else:
        augmentation_params = dict(DEFAULT_AUGMENTATION_PARAMETERS, **augm)

    if quasi_random_generator is None:
        quasi_random_generator = quasi_random.scrambled_halton_sequence_generator(dimension=len(augmentation_params),
                                                                                  permutation='Braaten-Weller')
    res = dict()
    try:
        sample = quasi_random_generator.next()
    except ValueError:
        quasi_random_generator = quasi_random.scrambled_halton_sequence_generator(dimension=len(augmentation_params),
                                                                                  permutation='Braaten-Weller')
        sample = quasi_random_generator.next()

    for rand, (key, (a, b)) in izip(sample, augmentation_params.iteritems()):
        #res[key] = config().rng.uniform(a,b)
        res[key] = a + rand*(b-a)
    return res


def put_in_the_middle(target_tensor, data_tensor, pad_better=False, is_padded=None):
    """
    put data_sensor with arbitrary number of dimensions in the middle of target tensor.
    if data_tensor is bigger, data is cut off
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
    if is_padded is not None:
        is_padded[:] = True
        is_padded[target_indices] = False
    if pad_better:
        if target_indices[0].start:
            for i in xrange(0, target_indices[0].start):
                target_tensor[i] = data_tensor[0]
        if target_indices[0].stop:
            for i in xrange(target_indices[0].stop, len(target_tensor)):
                target_tensor[i] = data_tensor[-1]


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


def _make_4d_tensor(tensors):
    """
    Input: list of 3d tensors with a different first dimension.
    Output: 4d tensor
    """
    max_frames = max([t.shape[0] for t in tensors])
    min_frames = min([t.shape[0] for t in tensors])
    # If all dimensions are equal, just make an array out of it
    if min_frames == max_frames:
        return np.array(tensors)
    # Otherwise, we need to do it manually
    else:
        res = np.zeros((len(tensors), max_frames, tensors[0].shape[1], tensors[0].shape[2]))
        for i, t in enumerate(tensors):
            nr_padding_frames = max_frames - len(t)
            res[i] = np.vstack([t] + [t[:1]]*nr_padding_frames)
        return res


def preprocess_normscale(patient_data, result, index, augment=True,
                         metadata=None,
                         normscale_resize_and_augment_function=normscale_resize_and_augment,
                         testaug=False):
    """Normalizes scale and augments the data.

    Args:
        patient_data: the data to be preprocessed.
        result: dict to store the result in.
        index: index indicating in which slot the result dict the data
            should go.
        augment: flag indicating wheter augmentation is needed.
        metadata: metadata belonging to the patient data.
    """
    if augment:
        if testaug:
            augmentation_params = sample_test_augmentation_parameters()
        else:
            augmentation_params = sample_augmentation_parameters()
    else:
        augmentation_params = None

    zoom_factor = None

    # Iterate over different sorts of data
    for tag, data in patient_data.iteritems():
        if tag in metadata:
            metadata_tag = metadata[tag]
        desired_shape = result[tag][index].shape

        cleaning_processes = getattr(config(), 'cleaning_processes', [])
        cleaning_processes_post = getattr(config(), 'cleaning_processes_post', [])

        if tag.startswith("sliced:data:singleslice"):
            # Cleaning data before extracting a patch
            data = clean_images(
                [patient_data[tag]], metadata=metadata_tag,
                cleaning_processes=cleaning_processes)

            # Augment and extract patch
            # Decide which roi to use.
            shift_center = (None, None)
            if getattr(config(), 'use_hough_roi', False):
                shift_center = metadata_tag["hough_roi"]

            patient_3d_tensor = normscale_resize_and_augment_function(
                data, output_shape=desired_shape[-2:],
                augment=augmentation_params,
                pixel_spacing=metadata_tag["PixelSpacing"],
                shift_center=shift_center[::-1])[0]

            if augmentation_params is not None:
                zoom_factor = augmentation_params["zoom_x"] * augmentation_params["zoom_y"]
            else:
                zoom_factor = 1.0
                
            # Clean data further
            patient_3d_tensor = clean_images(
                patient_3d_tensor, metadata=metadata_tag,
                cleaning_processes=cleaning_processes_post)

            if "area_per_pixel:sax" in result:
                raise NotImplementedError()

            if augmentation_params and not augmentation_params.get("change_brightness", 0) == 0:
                patient_3d_tensor = augment_brightness(patient_3d_tensor, augmentation_params["change_brightness"])

            put_in_the_middle(result[tag][index], patient_3d_tensor, True)


        elif tag.startswith("sliced:data:randomslices"):
            # Clean each slice separately
            data = [
                clean_images([slicedata], metadata=metadata, cleaning_processes=cleaning_processes)[0]
                for slicedata, metadata in zip(data, metadata_tag)]

            # Augment and extract patches
            shift_centers = [(None, None)] * len(data)
            if getattr(config(), 'use_hough_roi', False):
                shift_centers = [m["hough_roi"] for m in metadata_tag]

            patient_3d_tensors = [
                normscale_resize_and_augment_function(
                    [slicedata], output_shape=desired_shape[-2:],
                    augment=augmentation_params,
                    pixel_spacing=metadata["PixelSpacing"],
                    shift_center=shift_center[::-1])[0]
                for slicedata, metadata, shift_center in zip(data, metadata_tag, shift_centers)]
            if augmentation_params is not None:
                zoom_factor = augmentation_params["zoom_x"] * augmentation_params["zoom_y"]
            else:
                zoom_factor = 1.0

            # Clean data further
            patient_3d_tensors = [
                clean_images([patient_3d_tensor], metadata=metadata, cleaning_processes=cleaning_processes_post)[0]
                for patient_3d_tensor, metadata in zip(patient_3d_tensors, metadata_tag)]

            patient_4d_tensor = _make_4d_tensor(patient_3d_tensors)

            if augmentation_params and not augmentation_params.get("change_brightness", 0) == 0:
                patient_4d_tensor = augment_brightness(patient_4d_tensor, augmentation_params["change_brightness"])

            if "area_per_pixel:sax" in result:
                raise NotImplementedError()

            put_in_the_middle(result[tag][index], patient_4d_tensor, True)

        elif tag.startswith("sliced:data:sax:locations"):
            pass  # will be filled in by the next one
        elif tag.startswith("sliced:data:sax:is_not_padded"):
            pass  # will be filled in by the next one
        elif tag.startswith("sliced:data:sax"):
            # step 1: sort (data, metadata_tag) with slice_location_finder
            slice_locations, sorted_indices, sorted_distances = slice_location_finder({i: metadata for i,metadata in enumerate(metadata_tag)})

            data = [data[idx] for idx in sorted_indices]
            metadata_tag = [metadata_tag[idx] for idx in sorted_indices]

            slice_locations = np.array([slice_locations[idx]["relative_position"] for idx in sorted_indices])
            slice_locations = slice_locations - (slice_locations[-1] + slice_locations[0])/2.0

            data = [
                clean_images([slicedata], metadata=metadata, cleaning_processes=cleaning_processes)[0]
                for slicedata, metadata in zip(data, metadata_tag)]

            # Augment and extract patches
            shift_centers = [(None, None)] * len(data)
            if getattr(config(), 'use_hough_roi', False):
                shift_centers = [m["hough_roi"] for m in metadata_tag]

            patient_3d_tensors = [
                normscale_resize_and_augment_function(
                    [slicedata], output_shape=desired_shape[-2:],
                    augment=augmentation_params,
                    pixel_spacing=metadata["PixelSpacing"],
                    shift_center=shift_center[::-1])[0]
                for slicedata, metadata, shift_center in zip(data, metadata_tag, shift_centers)]

            if augmentation_params is not None:
                zoom_factor = augmentation_params["zoom_x"] * augmentation_params["zoom_y"]
            else:
                zoom_factor = 1.0

            # Clean data further
            patient_3d_tensors = [
                clean_images([patient_3d_tensor], metadata=metadata, cleaning_processes=cleaning_processes_post)[0]
                for patient_3d_tensor, metadata in zip(patient_3d_tensors, metadata_tag)]

            patient_4d_tensor = _make_4d_tensor(patient_3d_tensors)

            if augmentation_params and not augmentation_params.get("change_brightness", 0) == 0:
                patient_4d_tensor = augment_brightness(patient_4d_tensor, augmentation_params["change_brightness"])

            # Augment sax order
            if augmentation_params and augmentation_params.get("flip_sax", 0) > 0.5:
                patient_4d_tensor = patient_4d_tensor[::-1]
                slice_locations = slice_locations[::-1]

            # Put data (images and metadata) in right location
            put_in_the_middle(result[tag][index], patient_4d_tensor, True)

            if "sliced:data:sax:locations" in result:
                eps_location = 1e-7
                is_padded = np.array([False]*len(result["sliced:data:sax:locations"][index]))
                put_in_the_middle(result["sliced:data:sax:locations"][index], slice_locations + eps_location, True, is_padded)

            if "sliced:data:sax:distances" in result:
                eps_location = 1e-7
                sorted_distances.append(0.0)  # is easier for correct padding
                is_padded = np.array([False]*len(result["sliced:data:sax:distances"][index]))
                put_in_the_middle(result["sliced:data:sax:distances"][index], np.array(sorted_distances) + eps_location, True, is_padded)

            if "sliced:data:sax:is_not_padded" in result:
                result["sliced:data:sax:is_not_padded"][index] = np.logical_not(is_padded)



        elif tag.startswith("sliced:data:chanzoom:2ch"):
            # step 1: sort (data, metadata_tag) with slice_location_finder
            slice_locations, sorted_indices, sorted_distances = slice_location_finder({i: metadata for i,metadata in enumerate(metadata_tag[2])})

            top_slice_metadata = metadata_tag[2][sorted_indices[0]]
            bottom_slice_metadata = metadata_tag[2][sorted_indices[-1]]

            ch2_metadata = metadata_tag[1]
            ch4_metadata = metadata_tag[0]

            trf_2ch, trf_4ch = get_chan_transformations(
                ch2_metadata=ch2_metadata,
                ch4_metadata=ch4_metadata,
                top_point_metadata = top_slice_metadata,
                bottom_point_metadata = bottom_slice_metadata,
                output_width=desired_shape[-1]
                )

            ch4_3d_patient_tensor, ch2_3d_patient_tensor = [], []
            ch4_data = data[0]
            ch2_data = data[1]
            if ch4_data is None and ch2_data is not None:
                ch4_data = ch2_data
                ch4_metadata = ch2_metadata
            if ch2_data is None and ch4_data is not None:
                ch2_data = ch4_data
                ch2_metadata = ch4_metadata

            for ch, ch_result, transform, metadata in [(ch4_data, ch4_3d_patient_tensor, trf_4ch, ch4_metadata),
                                                        (ch2_data, ch2_3d_patient_tensor, trf_2ch, ch2_metadata)]:
                tform_shift_center, tform_shift_uncenter = build_center_uncenter_transforms(desired_shape[-2:])
                zoom_factor = np.sqrt(np.abs(np.linalg.det(transform.params[:2,:2])) * np.prod(metadata["PixelSpacing"]))
                normalise_zoom_transform = build_augmentation_transform(zoom_x=zoom_factor, zoom_y=zoom_factor)
                if augmentation_params:
                    augment_tform = build_augmentation_transform(**augmentation_params)
                    total_tform = tform_shift_uncenter + augment_tform + normalise_zoom_transform + tform_shift_center + transform
                else:
                    total_tform = tform_shift_uncenter + normalise_zoom_transform + tform_shift_center + transform

                ch_result[:] = [fast_warp(c, total_tform, output_shape=desired_shape[-2:]) for c in ch]
                # print "zoom factor:", zoom_factor

            if augmentation_params is not None:
                zoom_factor = augmentation_params["zoom_x"] * augmentation_params["zoom_y"]
            else:
                zoom_factor = 1.0
            # Clean data further
            ch4_3d_patient_tensor = clean_images(np.array([ch4_3d_patient_tensor]), metadata=ch4_metadata, cleaning_processes=cleaning_processes_post)[0]
            ch2_3d_patient_tensor = clean_images(np.array([ch2_3d_patient_tensor]), metadata=ch2_metadata, cleaning_processes=cleaning_processes_post)[0]

            # Put data (images and metadata) in right location
            put_in_the_middle(result["sliced:data:chanzoom:2ch"][index], ch2_3d_patient_tensor, True)
            put_in_the_middle(result["sliced:data:chanzoom:4ch"][index], ch4_3d_patient_tensor, True)

        elif tag.startswith("sliced:data:shape"):
            raise NotImplementedError()

        elif tag.startswith("sliced:data"):
            # put time dimension first, then axis dimension
            data = clean_images(patient_data[tag], metadata=metadata_tag)
            patient_4d_tensor, zoom_ratios = resize_and_augment(data, output_shape=desired_shape[-2:], augment=augmentation_parameters)
            if "area_per_pixel:sax" in result:
                result["area_per_pixel:sax"][index] = zoom_ratios[0] * np.prod(metadata_tag[0]["PixelSpacing"])

            if "noswitch" not in tag:
                patient_4d_tensor = np.swapaxes(patient_4d_tensor,1,0)

            put_in_the_middle(result[tag][index], patient_4d_tensor)

        elif tag.startswith("sliced:meta:all"):
            # TODO: this probably doesn't work very well yet
            result[tag][index] = patient_data[tag]

        elif tag.startswith("sliced:meta:PatientSex"):
            result[tag][index][0] = -1. if patient_data[tag]=='M' else 1.

        elif tag.startswith("sliced:meta:PatientAge"):
            number, letter = patient_data[tag][:3], patient_data[tag][-1]
            letter_rescale_factors = {'D': 365.25, 'W': 52.1429, 'M': 12., 'Y': 1.}
            result[tag][index][0] = float(patient_data[tag][:3]) / letter_rescale_factors[letter]

    if augmentation_params and zoom_factor:
        label_correction_function = lambda x: x * zoom_factor
        classification_correction_function = lambda x: utils.zoom_array(x, 1./zoom_factor)
        return label_correction_function, classification_correction_function
    else:
        return lambda x: x, lambda x: x


def preprocess_with_augmentation(patient_data, result, index, augment=True, metadata=None, testaug=False):
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
        metadata_tag = metadata[tag]
        desired_shape = result[tag][index].shape
        # try to fit data into the desired shape
        if tag.startswith("sliced:data:singleslice"):
            cleaning_processes = getattr(config(), 'cleaning_processes', [])
            data = clean_images(
                [patient_data[tag]], metadata=metadata_tag,
                cleaning_processes=cleaning_processes)
            patient_4d_tensor, zoom_ratios = resize_and_augment(data, output_shape=desired_shape[-2:], augment=augmentation_parameters)[0]
            if "area_per_pixel:sax" in result:
                result["area_per_pixel:sax"][index] = zoom_ratios[0] * np.prod(metadata_tag["PixelSpacing"])

            put_in_the_middle(result[tag][index], patient_4d_tensor)
        elif tag.startswith("sliced:data"):
            # put time dimension first, then axis dimension
            data = clean_images(patient_data[tag], metadata=metadata_tag)
            patient_4d_tensor, zoom_ratios = resize_and_augment(data, output_shape=desired_shape[-2:], augment=augmentation_parameters)
            if "area_per_pixel:sax" in result:
                result["area_per_pixel:sax"][index] = zoom_ratios[0] * np.prod(metadata_tag[0]["PixelSpacing"])

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


def clean_images(data, metadata, cleaning_processes):
    """
    clean up 4d-tensor of imdata consistently (fix contrast, move upside up, etc...)
    :param data:
    :return:
    """
    for process in cleaning_processes:
        data = process(data, metadata)
    return data


def normalize_contrast(imdata, metadata=None, percentiles=(5.0,95.0)):
    lp, hp = percentiles
    flat_data = np.concatenate([i.flatten() for i in imdata]).flatten()
    high = np.percentile(flat_data, hp)
    low  = np.percentile(flat_data, lp)
    for i in xrange(len(imdata)):
        image = imdata[i]
        image = 1.0 * (image - low) / (high - low)
        image = np.clip(image, 0.0, 1.0)
        imdata[i] = image

    return imdata


def normalize_contrast_zmuv(imdata, metadata=None, z=2):
    flat_data = np.concatenate([i.flatten() for i in imdata]).flatten()
    mean = np.mean(flat_data)
    std = np.std(flat_data)
    for i in xrange(len(imdata)):
        image = imdata[i]
        image = ((image - mean) / (2 * std * z) + 0.5)
        image = np.clip(image, -0.0, 1.0)
        imdata[i] = image

    return imdata


def set_upside_up(data, metadata=None):
    out_data = []
    for idx, dslice in enumerate(data):
        out_data.append(set_upside_up_slice(dslice, metadata))
    return out_data


_TAG_ROI_UPSIDEUP = 'ROI_UPSIDEUP'
def set_upside_up_slice(dslice, metadata=None, do_flip=False):
    # turn upside up
    F = np.array(metadata["ImageOrientationPatient"]).reshape((2, 3))

    f_1 = F[1, :] / np.linalg.norm(F[1, :])
    f_2 = F[0, :] / np.linalg.norm(F[0, :])

    x_e = np.array([1, 0, 0])
    y_e = np.array([0, 1, 0])

    if abs(np.dot(y_e, f_1)) >= abs(np.dot(y_e, f_2)):
        out_data = np.transpose(dslice, (0, 2, 1))
        out_roi = list(metadata["hough_roi"][::-1])
        f_1, f_2 = f_2, f_1
    else:
        out_data = dslice
        out_roi = list(metadata["hough_roi"])

    if np.dot(y_e, f_1) < 0 and do_flip:
        # Flip vertically
        out_data = out_data[:, ::-1, :]
        if out_roi[0]: out_roi[0] = 1 - out_roi[0]

    if np.dot(x_e, f_2) < 0 and do_flip:
        # Flip horizontally
        out_data = out_data[:, :, ::-1]
        if out_roi[1]: out_roi[1] = 1 - out_roi[1]

    if not _TAG_ROI_UPSIDEUP in metadata:
        metadata["hough_roi"] = tuple(out_roi)
        metadata[_TAG_ROI_UPSIDEUP] = True

    return out_data


def slice_location_finder(metadata_dict):
    """
    :param metadata_dict: dict with arbitrary keys, and metadata values
    :return: dict with "relative_position" and "middle_pixel_position" (and others)
    """
    datadict = dict()

    for key, metadata in metadata_dict.iteritems():
        #d1 = all_data['data']
        d2 = metadata
        image_orientation = [float(i) for i in metadata["ImageOrientationPatient"]]
        image_position = [float(i) for i in metadata["ImagePositionPatient"]]
        pixel_spacing = [float(i) for i in metadata["PixelSpacing"]]
        datadict[key] = {
            "orientation": image_orientation,
            "position": image_position,
            "pixel_spacing": pixel_spacing,
            "rows": int(d2["Rows"]),
            "columns": int(d2["Columns"]),
        }

    for key, data in datadict.iteritems():
        # calculate value of middle pixel
        F = np.array(data["orientation"]).reshape( (2,3) )
        pixel_spacing = data["pixel_spacing"]
        i,j = data["columns"] / 2.0, data["rows"] / 2.0  # reversed order, as per http://nipy.org/nibabel/dicom/dicom_orientation.html
        im_pos = np.array([[i*pixel_spacing[0],j*pixel_spacing[1]]],dtype='float32')
        pos = np.array(data["position"]).reshape((1,3))
        position = np.dot(im_pos, F) + pos
        data["middle_pixel_position"] = position[0,:]

    # find the keys of the 2 points furthest away from each other
    if len(datadict)<=1:
        for key, data in datadict.iteritems():
            data["relative_position"] = 0.0
    else:
        max_dist = -1.0
        max_dist_keys = []
        for key1, data1 in datadict.iteritems():
            for key2, data2 in datadict.iteritems():
                if key1==key2:
                    continue
                p1 = data1["middle_pixel_position"]
                p2 = data2["middle_pixel_position"]
                distance = np.sqrt(np.sum((p1-p2)**2))
                if distance>max_dist:
                    max_dist_keys = [key1, key2]
                    max_dist = distance
        # project the others on the line between these 2 points
        # sort the keys, so the order is more or less the same as they were
        max_dist_keys.sort()
        p_ref1 = datadict[max_dist_keys[0]]["middle_pixel_position"]
        p_ref2 = datadict[max_dist_keys[1]]["middle_pixel_position"]
        v1 = p_ref2-p_ref1
        v1 = v1 / np.linalg.norm(v1)
        for key, data in datadict.iteritems():
            v2 = data["middle_pixel_position"]-p_ref1
            scalar = np.inner(v1, v2)
            data["relative_position"] = scalar

    sorted_indices = [key for key in sorted(datadict.iterkeys(), key=lambda x: datadict[x]["relative_position"])]

    sorted_distances = []
    for i in xrange(len(sorted_indices)-1):
        res = []
        for d1, d2 in [(datadict[sorted_indices[i]], datadict[sorted_indices[i+1]]),
                       (datadict[sorted_indices[i+1]], datadict[sorted_indices[i]])]:
            F = np.array(d1["orientation"]).reshape( (2,3) )
            n = np.cross(F[0,:], F[1,:])
            n = n/np.sqrt(np.sum(n*n))
            p = d2["middle_pixel_position"] - d1["position"]
            distance = np.abs(np.sum(n*p))
            res.append(distance)
        sorted_distances.append(np.mean(res))

    return datadict, sorted_indices, sorted_distances




def orthogonal_projection_on_slice(percentual_coordinate, source_metadata, target_metadata):
    point = np.array([[percentual_coordinate[0]],
                      [percentual_coordinate[1]],
                      [0],
                      [1]])
    image_size = [source_metadata["Rows"], source_metadata["Columns"]]
    point = np.dot(np.array(  [[image_size[0],0,0,0],
                               [0,image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = source_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[pixel_spacing[0],0,0,0],
                               [0,pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    Fa = np.array(source_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    posa = source_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[Fa[0,0],Fa[1,0],0,posa[0]],
                               [Fa[0,1],Fa[1,1],0,posa[1]],
                               [Fa[0,2],Fa[1,2],0,posa[2]],
                               [0,0,0,1]]), point)
    posb = target_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[1,0,0,-posb[0]],
                               [0,1,0,-posb[1]],
                               [0,0,1,-posb[2]],
                               [0,0,0,1]]), point)
    Fb = np.array(target_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    ff0 = np.sqrt(np.sum(Fb[0,:]*Fb[0,:]))
    ff1 = np.sqrt(np.sum(Fb[1,:]*Fb[1,:]))

    point = np.dot(np.array(  [[Fb[0,0]/ff0,Fb[0,1]/ff0,Fb[0,2]/ff0,0],
                               [Fb[1,0]/ff1,Fb[1,1]/ff1,Fb[1,2]/ff1,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = target_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[1./pixel_spacing[0],0,0,0],
                               [0,1./pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    image_size = [target_metadata["Rows"], target_metadata["Columns"]]
    point = np.dot(np.array(  [[1./image_size[0],0,0,0],
                               [0,1./image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    return point[:2,0]  # percentual coordinate as well


def patient_coor_from_slice(percentual_coordinate, source_metadata):
    point = np.array([[percentual_coordinate[0]],
                      [percentual_coordinate[1]],
                      [0],
                      [1]])
    image_size = [source_metadata["Rows"], source_metadata["Columns"]]
    point = np.dot(np.array(  [[image_size[0],0,0,0],
                               [0,image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = source_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[pixel_spacing[0],0,0,0],
                               [0,pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    Fa = np.array(source_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    posa = source_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[Fa[0,0],Fa[1,0],0,posa[0]],
                               [Fa[0,1],Fa[1,1],0,posa[1]],
                               [Fa[0,2],Fa[1,2],0,posa[2]],
                               [0,0,0,1]]), point)

    return point[:3,0]  # patient coordinate


def point_projection_on_slice(point, target_metadata):
    point = np.array([[point[0]],
                      [point[1]],
                      [point[2]],
                      [1]])
    posb = target_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[1,0,0,-posb[0]],
                               [0,1,0,-posb[1]],
                               [0,0,1,-posb[2]],
                               [0,0,0,1]]), point)
    Fb = np.array(target_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    ff0 = np.sqrt(np.sum(Fb[0,:]*Fb[0,:]))
    ff1 = np.sqrt(np.sum(Fb[1,:]*Fb[1,:]))

    point = np.dot(np.array(  [[Fb[0,0]/ff0,Fb[0,1]/ff0,Fb[0,2]/ff0,0],
                               [Fb[1,0]/ff1,Fb[1,1]/ff1,Fb[1,2]/ff1,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = target_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[1./pixel_spacing[0],0,0,0],
                               [0,1./pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    return point[:2,0]  # percentual coordinate as well



def get_chan_transformations(ch2_metadata=None,
                             ch4_metadata=None,
                             top_point_metadata=None,
                             bottom_point_metadata=None,
                             output_width = 100):

    has_both_chans = False
    if ch2_metadata is None and ch4_metadata is None:
        raise "Need at least one of these slices"
    elif ch2_metadata and ch4_metadata is None:
        ch4_metadata = ch2_metadata
    elif ch4_metadata and ch2_metadata is None:
        ch2_metadata = ch4_metadata
    else:
        has_both_chans = True


    F2 = np.array(ch2_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    F4 = np.array(ch4_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]

    n2 = np.cross(F2[0,:], F2[1,:])
    n4 = np.cross(F4[0,:], F4[1,:])

    b2 = np.sum(n2 * np.array(ch2_metadata["ImagePositionPatient"]))
    b4 = np.sum(n4 * np.array(ch4_metadata["ImagePositionPatient"]))

    # find top and bottom of my view
    top_point = patient_coor_from_slice(top_point_metadata["hough_roi"], top_point_metadata)
    bottom_point = patient_coor_from_slice(bottom_point_metadata["hough_roi"], bottom_point_metadata)

    # if it has both chan's: middle line is the common line!
    if has_both_chans:
        F5 = np.cross(n2, n4)
        A = np.array([n2, n4])
        b = np.array([b2, b4])
        #print A, b
        P, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print P, rnorm, rank, s

        # find top and bottom on the line
        A = np.array([F5]).T
        b = np.array(top_point)
        #print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print sc, rnorm, rank, s

        top_point = sc[0] * F5 + P

        A = np.array([F5]).T
        b = np.array(bottom_point)
        #print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print sc, rnorm, rank, s

        bottom_point = sc[0] * F5 + P

    ## FIND THE affine transformation ch2 needs:

    ch2_top_point = point_projection_on_slice(top_point, ch2_metadata)
    ch2_bottom_point = point_projection_on_slice(bottom_point, ch2_metadata)
    n = np.array([ch2_bottom_point[1] - ch2_top_point[1], ch2_top_point[0] - ch2_bottom_point[0]])
    ch2_third_point = ch2_top_point + n/2

    A = np.array([[ch2_top_point[0], ch2_top_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_top_point[0], ch2_top_point[1], 1],
                  [ch2_bottom_point[0], ch2_bottom_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_bottom_point[0], ch2_bottom_point[1], 1],
                  [ch2_third_point[0], ch2_third_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_third_point[0], ch2_third_point[1], 1],])
    b = np.array([0,0.5*output_width,output_width,0.5*output_width,0,0])
    #print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A,b)
    #print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [    0,     0,    1]]))
    ch2_form_fix  = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    # same for ch4
    ch4_top_point = point_projection_on_slice(top_point, ch4_metadata)
    ch4_bottom_point = point_projection_on_slice(bottom_point, ch4_metadata)
    n = np.array([ch4_bottom_point[1] - ch4_top_point[1], ch4_top_point[0] - ch4_bottom_point[0]])
    ch4_third_point = ch4_top_point + n/2

    A = np.array([[ch4_top_point[0], ch4_top_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_top_point[0], ch4_top_point[1], 1],
                  [ch4_bottom_point[0], ch4_bottom_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_bottom_point[0], ch4_bottom_point[1], 1],
                  [ch4_third_point[0], ch4_third_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_third_point[0], ch4_third_point[1], 1],])
    b = np.array([0,0.5*output_width,output_width,0.5*output_width,0,0])
    #print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A,b)
    #print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [    0,     0,    1]]))
    ch4_form_fix  = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    return ch2_form_fix, ch4_form_fix

def augment_brightness(patient_tensor, brightness_adjustment):
#    print "augmenting", brightness_adjustment
    return np.clip(patient_tensor + brightness_adjustment * np.mean(patient_tensor), 0, 1)


def orthogonal_projection_on_slice(percentual_coordinate, source_metadata, target_metadata):
    point = np.array([[percentual_coordinate[0]],
                      [percentual_coordinate[1]],
                      [0],
                      [1]])
    image_size = [source_metadata["Rows"], source_metadata["Columns"]]
    point = np.dot(np.array(  [[image_size[0],0,0,0],
                               [0,image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = source_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[pixel_spacing[0],0,0,0],
                               [0,pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    Fa = np.array(source_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    posa = source_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[Fa[0,0],Fa[1,0],0,posa[0]],
                               [Fa[0,1],Fa[1,1],0,posa[1]],
                               [Fa[0,2],Fa[1,2],0,posa[2]],
                               [0,0,0,1]]), point)
    posb = target_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[1,0,0,-posb[0]],
                               [0,1,0,-posb[1]],
                               [0,0,1,-posb[2]],
                               [0,0,0,1]]), point)
    Fb = np.array(target_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    ff0 = np.sqrt(np.sum(Fb[0,:]*Fb[0,:]))
    ff1 = np.sqrt(np.sum(Fb[1,:]*Fb[1,:]))

    point = np.dot(np.array(  [[Fb[0,0]/ff0,Fb[0,1]/ff0,Fb[0,2]/ff0,0],
                               [Fb[1,0]/ff1,Fb[1,1]/ff1,Fb[1,2]/ff1,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = target_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[1./pixel_spacing[0],0,0,0],
                               [0,1./pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    image_size = [target_metadata["Rows"], target_metadata["Columns"]]
    point = np.dot(np.array(  [[1./image_size[0],0,0,0],
                               [0,1./image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    return point[:2,0]  # percentual coordinate as well


def patient_coor_from_slice(percentual_coordinate, source_metadata):
    point = np.array([[percentual_coordinate[0]],
                      [percentual_coordinate[1]],
                      [0],
                      [1]])
    image_size = [source_metadata["Rows"], source_metadata["Columns"]]
    point = np.dot(np.array(  [[image_size[0],0,0,0],
                               [0,image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = source_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[pixel_spacing[0],0,0,0],
                               [0,pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    Fa = np.array(source_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    posa = source_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[Fa[0,0],Fa[1,0],0,posa[0]],
                               [Fa[0,1],Fa[1,1],0,posa[1]],
                               [Fa[0,2],Fa[1,2],0,posa[2]],
                               [0,0,0,1]]), point)

    return point[:3,0]  # patient coordinate


def point_projection_on_slice(point, target_metadata):
    point = np.array([[point[0]],
                      [point[1]],
                      [point[2]],
                      [1]])
    posb = target_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[1,0,0,-posb[0]],
                               [0,1,0,-posb[1]],
                               [0,0,1,-posb[2]],
                               [0,0,0,1]]), point)
    Fb = np.array(target_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    ff0 = np.sqrt(np.sum(Fb[0,:]*Fb[0,:]))
    ff1 = np.sqrt(np.sum(Fb[1,:]*Fb[1,:]))

    point = np.dot(np.array(  [[Fb[0,0]/ff0,Fb[0,1]/ff0,Fb[0,2]/ff0,0],
                               [Fb[1,0]/ff1,Fb[1,1]/ff1,Fb[1,2]/ff1,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = target_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[1./pixel_spacing[0],0,0,0],
                               [0,1./pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    return point[:2,0]  # percentual coordinate as well



def get_chan_transformations(ch2_metadata=None,
                             ch4_metadata=None,
                             top_point_metadata=None,
                             bottom_point_metadata=None,
                             output_width = 100):

    has_both_chans = False
    if ch2_metadata is None and ch4_metadata is None:
        raise "Need at least one of these slices"
    elif ch2_metadata and ch4_metadata is None:
        ch4_metadata = ch2_metadata
    elif ch4_metadata and ch2_metadata is None:
        ch2_metadata = ch4_metadata
    else:
        has_both_chans = True


    F2 = np.array(ch2_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    F4 = np.array(ch4_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]

    n2 = np.cross(F2[0,:], F2[1,:])
    n4 = np.cross(F4[0,:], F4[1,:])

    b2 = np.sum(n2 * np.array(ch2_metadata["ImagePositionPatient"]))
    b4 = np.sum(n4 * np.array(ch4_metadata["ImagePositionPatient"]))

    # find top and bottom of my view
    top_point = patient_coor_from_slice(top_point_metadata["hough_roi"], top_point_metadata)
    bottom_point = patient_coor_from_slice(bottom_point_metadata["hough_roi"], bottom_point_metadata)

    # if it has both chan's: middle line is the common line!
    if has_both_chans:
        F5 = np.cross(n2, n4)
        A = np.array([n2, n4])
        b = np.array([b2, b4])
        #print A, b
        P, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print P, rnorm, rank, s

        # find top and bottom on the line
        A = np.array([F5]).T
        b = np.array(top_point)
        #print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print sc, rnorm, rank, s

        top_point = sc[0] * F5 + P

        A = np.array([F5]).T
        b = np.array(bottom_point)
        #print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A,b)
        #print sc, rnorm, rank, s

        bottom_point = sc[0] * F5 + P

    ## FIND THE affine transformation ch2 needs:

    ch2_top_point = point_projection_on_slice(top_point, ch2_metadata)
    ch2_bottom_point = point_projection_on_slice(bottom_point, ch2_metadata)
    n = np.array([ch2_bottom_point[1] - ch2_top_point[1], ch2_top_point[0] - ch2_bottom_point[0]])
    ch2_third_point = ch2_top_point + n/2

    A = np.array([[ch2_top_point[0], ch2_top_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_top_point[0], ch2_top_point[1], 1],
                  [ch2_bottom_point[0], ch2_bottom_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_bottom_point[0], ch2_bottom_point[1], 1],
                  [ch2_third_point[0], ch2_third_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch2_third_point[0], ch2_third_point[1], 1],])
    b = np.array([0,0.5*output_width,output_width,0.5*output_width,0,0])
    #print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A,b)
    #print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [    0,     0,    1]]))
    ch2_form_fix  = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    # same for ch4
    ch4_top_point = point_projection_on_slice(top_point, ch4_metadata)
    ch4_bottom_point = point_projection_on_slice(bottom_point, ch4_metadata)
    n = np.array([ch4_bottom_point[1] - ch4_top_point[1], ch4_top_point[0] - ch4_bottom_point[0]])
    ch4_third_point = ch4_top_point + n/2

    A = np.array([[ch4_top_point[0], ch4_top_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_top_point[0], ch4_top_point[1], 1],
                  [ch4_bottom_point[0], ch4_bottom_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_bottom_point[0], ch4_bottom_point[1], 1],
                  [ch4_third_point[0], ch4_third_point[1], 1, 0, 0, 0 ],
                  [0, 0, 0, ch4_third_point[0], ch4_third_point[1], 1],])
    b = np.array([0,0.5*output_width,output_width,0.5*output_width,0,0])
    #print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A,b)
    #print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [    0,     0,    1]]))
    ch4_form_fix  = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    return ch2_form_fix, ch4_form_fix

def augment_brightness(patient_tensor, brightness_adjustment):
#    print "augmenting", brightness_adjustment
    return np.clip(patient_tensor + brightness_adjustment * np.mean(patient_tensor), 0, 1)