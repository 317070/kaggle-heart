"""Module responsible for loading and splitting the data.
"""
import cPickle as pickle
import glob
import itertools
import os
import random
import re

import numpy as np

import configuration
import disk_access
from paths import TEMP_FILES_PATH, PKL_TRAIN_DATA_PATH, TRAIN_PATIENT_IDS, TEST_PATIENT_IDS
from paths import PKL_TEST_DATA_PATH
import paths
import utils
import validation_set

_config = configuration.config  # shortcut for configuration


print "Loading data"

################
# Regular data #
################
# We don't load the regular data directly into memory, since it's too big.
_DATA_FOLDER = TEMP_FILES_PATH

_TRAIN_DATA_FOLDER = PKL_TRAIN_DATA_PATH
_TEST_DATA_FOLDER = PKL_TEST_DATA_PATH
_TRAIN_LABELS_PATH = os.path.join(TEMP_FILES_PATH, "train.pkl")

# TODO: don't make this hardcoded!
ALL_TRAIN_PATIENT_IDS = range(TRAIN_PATIENT_IDS[0], TRAIN_PATIENT_IDS[1] + 1)

def _extract_id_from_path(path):
    return int(re.search(r'/(\d+)/', path).group(1))

def _extract_slice_id_from_path(path):
    return int(re.search(r'_(\d+).pkl', path).group(1))


def _find_patient_folders(root_folder):
    """Finds and sorts all patient folders.
    """
    patient_folder_format = os.path.join(root_folder, "*", "study")
    patient_folders = glob.glob(patient_folder_format)
    patient_folders.sort(key=_extract_id_from_path)
    return patient_folders


def _split_train_val(patient_folders):
    """Splits the patient folders into train and validation splits.
    """
    # Construct train and validation splits using default parameters
    if paths.SUBMISSION_NR == 1:
        print "Using proper validation set"
        validation_patients_indices = validation_set.get_cross_validation_indices(
            indices=ALL_TRAIN_PATIENT_IDS, validation_index=0)
    else:
        print "WARNING: no validation set!!"
        validation_patients_indices = [1]

    train_patients_indices = [i for i in ALL_TRAIN_PATIENT_IDS if i not in validation_patients_indices]

    train_patient_folders = [
        folder for folder in patient_folders
        if not _extract_id_from_path(folder) in validation_patients_indices]
    validation_patient_folders = [
        folder for folder in patient_folders
        if folder not in train_patient_folders]

    return (
        train_patient_folders, validation_patient_folders,
        validation_patients_indices, train_patients_indices)


def _load_file(path):
    with open(path, "r") as f:
        data = pickle.load(f)
    return data


def _construct_id_to_index_map(patient_folders):
    res = {}
    for set in patient_folders:
        for index in xrange(len(patient_folders[set])):
            id = _extract_id_from_path(patient_folders[set][index])
            res[id] = (set, index)
    return res


def _in_folder(folder):
    wildcard_file_path = os.path.join(folder, "*")
    files = sorted(glob.glob(wildcard_file_path))
    return files


# Find train patients and split off validation
train_patient_folders, validation_patient_folders, validation_patients_indices, train_patients_indices=(
    _split_train_val(_find_patient_folders(_TRAIN_DATA_FOLDER)))
# Find test patients
test_patient_folders = _find_patient_folders(_TEST_DATA_FOLDER)
test_patients_indices =  range(TEST_PATIENT_IDS[0], TEST_PATIENT_IDS[1] + 1)
# Aggregate in a dict
patient_folders = {
    "train": train_patient_folders,
    "validation": validation_patient_folders,
    "test": test_patient_folders,
}

def get_patient_id(folder):
    return _extract_id_from_path(folder)

id_to_index_map = _construct_id_to_index_map(patient_folders)
num_patients = {set:len(patient_folders[set]) for set in patient_folders}
NUM_TRAIN_PATIENTS = num_patients['train']
NUM_VALID_PATIENTS = num_patients['validation']
NUM_TEST_PATIENTS = num_patients['test']
NUM_PATIENTS = NUM_TRAIN_PATIENTS + NUM_VALID_PATIENTS + NUM_TEST_PATIENTS

# Load the labels
regular_labels = _load_file(_TRAIN_LABELS_PATH)


def filter_patient_folders():
    global train_patient_folders, validation_patient_folders, test_patient_folders,\
            NUM_TRAIN_PATIENTS, NUM_VALID_PATIENTS, NUM_TEST_PATIENTS, NUM_PATIENTS,\
            num_patients
    if not hasattr(_config(), 'filter_samples'):
        return

    for set, key in patient_folders.iteritems():
        key[:] = _config().filter_samples(key)
        num_patients[set]=len(key)

    if NUM_TRAIN_PATIENTS != num_patients['train']:
        print "WARNING: keeping only %d of %d train patients!" % (num_patients['train'], NUM_TRAIN_PATIENTS)
    if NUM_VALID_PATIENTS != num_patients['validation']:
        print "WARNING: keeping only %d of %d validation patients!" % (num_patients['validation'], NUM_VALID_PATIENTS)
    if NUM_TEST_PATIENTS != num_patients['test']:
        print "WARNING: keeping only %d of %d test patients!" % (num_patients['test'], NUM_TEST_PATIENTS)

    NUM_TRAIN_PATIENTS = num_patients['train']
    NUM_VALID_PATIENTS = num_patients['validation']
    NUM_TEST_PATIENTS = num_patients['test']
    NUM_PATIENTS = NUM_TRAIN_PATIENTS + NUM_VALID_PATIENTS + NUM_TEST_PATIENTS


##############
# Sunny data #
##############
# This small dataset is loaded into memory
#_SUNNY_DATA_PATH = os.path.join(_DATA_FOLDER, "pkl_annotated", "data.pkl")

#_sunny_data = _load_file(_SUNNY_DATA_PATH)
#num_sunny_images = len(_sunny_data["images"])

#_validation_sunny_indices = validation_set.get_cross_validation_indices(
#    indices=range(num_sunny_images))
#_train_sunny_indices = [
#    i for i in range(num_sunny_images) if i not in _validation_sunny_indices]

#sunny_train_images = np.array(_sunny_data["images"])[_train_sunny_indices]
#sunny_train_labels = np.array(_sunny_data["labels"])[_train_sunny_indices]
#sunny_validation_images = np.array(_sunny_data["images"])[_validation_sunny_indices]
#sunny_validation_labels = np.array(_sunny_data["labels"])[_validation_sunny_indices]
sunny_train_images = [None] * 1000
sunny_train_labels = [None] * 1000
sunny_validation_images = [None] * 1000
sunny_validation_labels = [None] * 1000

###########################
# Data form preprocessing #
###########################

_HOUGH_ROI_PATHS = (
    TEMP_FILES_PATH + 'pkl_train_slice2roi.pkl',
    TEMP_FILES_PATH + 'pkl_validate_slice2roi.pkl',)
_hough_rois = utils.merge_dicts(map(_load_file, _HOUGH_ROI_PATHS))


##################################
# Methods for accessing the data #
##################################

_METADATA_ENHANCED_TAG = "META_ENHANCED"
def _is_enhanced(metadatadict):
    return metadatadict.get(_METADATA_ENHANCED_TAG, False)


def _tag_enhanced(metadatadict, is_enhanced=True):
    metadatadict[_METADATA_ENHANCED_TAG] = is_enhanced


def _enhance_metadata(metadata, patient_id, slice_name):
    if _is_enhanced(metadata):
        return
    # Add hough roi metadata using relative coordinates
    roi_center = list(_hough_rois[str(patient_id)][slice_name]['roi_center'])
    if not roi_center == (None, None):
        roi_center[0] = float(roi_center[0]) / metadata['Rows']
        roi_center[1] = float(roi_center[1]) / metadata['Columns']
    metadata['hough_roi'] = tuple(roi_center)
    metadata['hough_roi_radii'] = _hough_rois[str(patient_id)][slice_name]['roi_radii']
    _tag_enhanced(metadata)


def get_patient_data(indices, wanted_input_tags, wanted_output_tags,
                     set="train", preprocess_function=None, testaug=False):
    """
    return a dict with the desired data matched to the required tags
    :param wanted_data_tags:
    :return:
    """

    def initialise_empty():
        """Initialise empty chunk
        """
        result = {
            "input": {},
            "output": {},
        }

        no_samples = _config().batch_size * _config().batches_per_chunk
        vector_size = (no_samples, )
        matrix_size = (no_samples, 600)

        OUTPUT_DATA_SIZE_TYPE = {
            "systole": (matrix_size, "float32"),
            "diastole": (matrix_size, "float32"),
            "average": (matrix_size, "float32"),
            "systole:onehot": (matrix_size, "float32"),
            "diastole:onehot": (matrix_size, "float32"),
            "systole:class_weight": (matrix_size, "float32"),
            "diastole:class_weight": (matrix_size, "float32"),
            "systole:value": (vector_size, "float32"),
            "diastole:value": (vector_size, "float32"),
            "patients": (vector_size, "int32"),
            "slices": (vector_size, "int32"),
            "area_per_pixel": (no_samples, ),
        }

        for tag in wanted_output_tags:
            if tag in OUTPUT_DATA_SIZE_TYPE:
                size, dtype = OUTPUT_DATA_SIZE_TYPE[tag]
                result["output"][tag] = np.zeros(size, dtype=dtype)

        for tag in wanted_input_tags:
            if tag in _config().data_sizes:
                chunk_shape = list(_config().data_sizes[tag])
                chunk_shape[0] = chunk_shape[0] * _config().batches_per_chunk
                chunk_shape = tuple(chunk_shape)
                result["input"][tag] = np.zeros(chunk_shape, dtype="float32")

        if "classification_correction_function" in wanted_output_tags:
            result["output"]["classification_correction_function"] = [lambda x:x] * no_samples

        return result

    result = initialise_empty()

    if set not in patient_folders:
        raise ValueError("Don't know the dataset %s" % set)
    folders = [
        patient_folders[set][i] for i in indices if 0<=i<num_patients[set]]

    # Iterate over folders
    for i, folder in enumerate(folders):
        # find the id of the current patient in the folder name (=safer)
        id = _extract_id_from_path(folder)

        files = _in_folder(folder)
        patient_result = dict()
        metadatas_result = dict()
        # function for loading and cleaning metadata. Only use the first frame
        def load_clean_metadata(f):
            m = utils.clean_metadata(disk_access.load_metadata_from_file(f)[0])
            pid = _extract_id_from_path(f)
            slicename = os.path.basename(f)
            _enhance_metadata(m, pid, slicename)
            return m

        # Iterate over input tags
        for tag in wanted_input_tags:
            if tag.startswith("sliced:data:singleslice"):
                if "4ch" in tag:
                    l = [sax for sax in files if "4ch" in sax]
                elif  "2ch" in tag:
                    l = [sax for sax in files if "2ch" in sax]
                else:
                    l = [sax for sax in files if "sax" in sax]
                if not l:
                    if hasattr(_config(), 'check_inputs') and _config().check_inputs:
                        print "Warning: patient %d has no images of this type" % id
                    continue
                if "middle" in tag:
                    # Sort sax files, based on the integer in their name
                    l.sort(key=lambda f: int(re.findall("\d+", os.path.basename(f))[0]))
                    f = l[len(l)/2]
                else:
                    f = random.choice(l)
                patient_result[tag] = disk_access.load_data_from_file(f)
                metadatas_result[tag] = load_clean_metadata(f)
                slice_id = _extract_slice_id_from_path(f)
                if "difference" in tag:
                    for j in xrange(patient_result[tag].shape[0]-1):
                        patient_result[tag][j] -= patient_result[tag][j+1]
                    patient_result[tag] = np.delete(patient_result[tag],-1,0)
            elif tag.startswith("sliced:data:chanzoom:4ch"):
                pass # done by the next one
            elif tag.startswith("sliced:data:chanzoom:2ch"):
                l_4ch = [sax for sax in files if "4ch" in sax]
                l_2ch = [sax for sax in files if "2ch" in sax]
                patient_result[tag] = [disk_access.load_data_from_file(l_4ch[0]) if l_4ch else None,
                                       disk_access.load_data_from_file(l_2ch[0]) if l_2ch else None]
                metadatas_result[tag] = [load_clean_metadata(l_4ch[0]) if l_4ch else None,
                                         load_clean_metadata(l_2ch[0]) if l_2ch else None,
                                         None]


                l = [sax for sax in files if "sax" in sax]
                metadatas_result[tag][2] = [load_clean_metadata(f) for f in l]

            elif tag.startswith("sliced:data:randomslices"):
                l = [sax for sax in files if "sax" in sax]
                nr_slices = result["input"][tag].shape[1]
                chosen_files = utils.pick_random(l, nr_slices)
                patient_result[tag] = [disk_access.load_data_from_file(f) for f in chosen_files]
                metadatas_result[tag] = [load_clean_metadata(f) for f in chosen_files]

            elif tag.startswith("sliced:data:sax:locations"):
                pass  # will be filled in by sliced:data:sax

            elif tag.startswith("sliced:data:sax:distances"):
                pass  # will be filled in by sliced:data:sax

            elif tag.startswith("sliced:data:sax:is_not_padded"):
                pass  # will be filled in by sliced:data:sax

            elif tag.startswith("sliced:data:sax:distances"):
                pass  # will be filled in by the next one

            elif tag.startswith("sliced:data:sax:distances"):
                pass  # will be filled in by the next one

            elif tag.startswith("sliced:data:sax"):
                patient_result[tag] = [disk_access.load_data_from_file(f) for f in files if "sax" in f]
                metadatas_result[tag] = [load_clean_metadata(f) for f in files if "sax" in f]

            elif tag.startswith("sliced:data:shape"):
                patient_result[tag] = [disk_access.load_data_from_file(f).shape for f in files]
                metadatas_result[tag] = [load_clean_metadata(f) for f in files if "sax" in f]

            elif tag.startswith("sliced:data"):
                patient_result[tag] = [disk_access.load_data_from_file(f) for f in files]
                metadatas_result[tag] = [load_clean_metadata(f) for f in files]

            elif tag.startswith("area_per_pixel"):
                patient_result[tag] = None  # they are filled in in preprocessing

            elif tag.startswith("sliced:meta:all"):
                # get the key used in the pickle
                key = tag[len("slided:meta:all:"):]
                patient_result[tag] = [disk_access.load_metadata_from_file(f)[0][key] for f in files]
            elif tag.startswith("sliced:meta"):
                # get the key used in the pickle
                key = tag[len("slided:meta:"):]
                metadata_field = disk_access.load_metadata_from_file(files[0])[0][key]
                patient_result[tag] = metadata_field
            # add others when needed

        label_correction_function, classification_correction_function = preprocess_function(patient_result, result=result["input"], index=i, metadata=metadatas_result, testaug=True)

        if "classification_correction_function" in wanted_output_tags:
            result["output"]["classification_correction_function"][i] = classification_correction_function

        # load the labels
        if "patients" in wanted_output_tags:
            result["output"]["patients"][i] = id

        if "slices" in wanted_output_tags:
            result["output"]["slices"][i] = slice_id

        # only read labels, when we actually have them
        if id in regular_labels[:, 0]:
            assert regular_labels[id-1, 0]==id
            V_systole = label_correction_function(regular_labels[id-1, 1])
            V_diastole = label_correction_function(regular_labels[id-1, 2])

            if "systole" in wanted_output_tags:
                result["output"]["systole"][i][int(np.ceil(V_systole)):] = 1.0
            if "diastole" in wanted_output_tags:
                result["output"]["diastole"][i][int(np.ceil(V_diastole)):] = 1.0
            if "average" in wanted_output_tags:
                result["output"]["average"][i][int(np.ceil((V_diastole + V_systole)/2.0)):] = 1.0

            if "systole:onehot" in wanted_output_tags:
                result["output"]["systole:onehot"][i][int(np.ceil(V_systole))] = 1.0
            if "diastole:onehot" in wanted_output_tags:
                result["output"]["diastole:onehot"][i][int(np.ceil(V_diastole))] = 1.0

            if "systole:value" in wanted_output_tags:
                result["output"]["systole:value"][i] = V_systole
            if "diastole:value" in wanted_output_tags:
                result["output"]["diastole:value"][i] = V_diastole

            if "systole:class_weight" in wanted_output_tags:
                result["output"]["systole:class_weight"][i] = utils.linear_weighted(V_systole)
            if "diastole:class_weight" in wanted_output_tags:
                result["output"]["diastole:class_weight"][i] = utils.linear_weighted(V_diastole)

        else:
            if set!="test":
                raise Exception("unknown patient in train or validation set")


    # Check if any of the inputs or outputs are still empty!
    if hasattr(_config(), 'check_inputs') and _config().check_inputs:
        for key, value in itertools.chain(result["input"].iteritems(), result["output"].iteritems()):
            if key=="classification_correction_function":
                continue
            if not np.any(value): #there are only zeros in value
                raise Exception("there is an empty value at key %s" % key)
            if not np.isfinite(value).all(): #there are NaN's or infinites somewhere
                print value
                raise Exception("there is a NaN at key %s" % key)

    return result


def get_sunny_patient_data(indices, set="train"):
    images = sunny_train_images
    labels = sunny_train_labels

    chunk_shape = list(_config().data_sizes["sunny"])
    chunk_shape[0] = chunk_shape[0] * _config().batches_per_chunk
    chunk_shape = tuple(chunk_shape)

    sunny_chunk = np.zeros(chunk_shape, dtype='float32')

    chunk_shape = list(_config().data_sizes["sunny"])
    chunk_shape[0] = chunk_shape[0] * _config().batches_per_chunk
    chunk_shape.remove(1)
    chunk_shape = tuple(chunk_shape)

    sunny_label_chunk = np.zeros(chunk_shape, dtype='float32')

    for k, idx in enumerate(indices):
        if indices[k]<len(images):
            img = images[indices[k]]
            lbl = labels[indices[k]]
            _config().sunny_preprocess(sunny_chunk[k], img, sunny_label_chunk[k], lbl)
        else:
            print "requesting out of bound sunny?"
            pass #zeros

    return {
        "input": {
            "sunny": sunny_chunk,
        },
        "output": {
            "segmentation": sunny_label_chunk,
        }
    }


def compute_nr_slices(patient_folder):
    files = _in_folder(patient_folder)
    return len([sax for sax in files if "sax" in sax])


def generate_train_batch(required_input_keys, required_output_keys):
    """Creates an iterator that returns train batches."""

    sunny_chunk_size = _config().sunny_batch_size * _config().batches_per_chunk
    chunk_size = _config().batch_size * _config().batches_per_chunk

    while True:
        result = {}
        input_keys_to_do = list(required_input_keys) #clone
        output_keys_to_do = list(required_output_keys) #clone
        if "sunny" in input_keys_to_do or "segmentation" in output_keys_to_do:
            indices = _config().rng.randint(0, len(sunny_train_images), sunny_chunk_size)
            sunny_patient_data = get_sunny_patient_data(indices, set="train")
            result = utils.merge(result, sunny_patient_data)
            input_keys_to_do.remove("sunny")
            output_keys_to_do.remove("segmentation")

        indices = _config().rng.randint(0, len(train_patient_folders), chunk_size)  #
        kaggle_data = get_patient_data(indices, input_keys_to_do, output_keys_to_do, set="train",
                                       preprocess_function=_config().preprocess_train)

        result = utils.merge(result, kaggle_data)

        yield result



def generate_validation_batch(required_input_keys, required_output_keys, set="validation"):
    # generate sunny data
    sunny_length = get_lenght_of_set(name="sunny", set=set)
    regular_length = get_lenght_of_set(name="regular", set=set)

    sunny_batches = int(np.ceil(sunny_length / float(_config().sunny_batch_size)))
    regular_batches = int(np.ceil(regular_length / float(_config().batch_size)))

    if "sunny" in required_input_keys or "segmentation" in required_output_keys:
        num_batches = max(sunny_batches, regular_batches)
    else:
        num_batches = regular_batches

    num_chunks = int(np.ceil(num_batches / float(_config().batches_per_chunk)))

    sunny_chunk_size = _config().batches_per_chunk * _config().sunny_batch_size
    regular_chunk_size = _config().batches_per_chunk * _config().batch_size

    for n in xrange(num_chunks):

        result = {}
        input_keys_to_do  = list(required_input_keys)  # clone
        output_keys_to_do = list(required_output_keys) # clone

        if "sunny" in input_keys_to_do or "segmentation" in output_keys_to_do:

            indices = range(n*sunny_chunk_size, (n+1)*sunny_chunk_size)

            sunny_patient_data = get_sunny_patient_data(indices, set="train")
            result = utils.merge(result, sunny_patient_data)
            input_keys_to_do.remove("sunny")
            output_keys_to_do.remove("segmentation")

        indices = range(n*regular_chunk_size, (n+1)*regular_chunk_size)
        kaggle_data = get_patient_data(indices, input_keys_to_do, output_keys_to_do, set=set,
                                       preprocess_function=_config().preprocess_validation)

        result = utils.merge(result, kaggle_data)

        yield result


def generate_test_batch(required_input_keys, required_output_keys, augmentation=False, set=None):
    if set is None:
        sets = ["validation", "test"]
    elif type(set) == list:
        sets = set
    else:
        sets=[set]

    input_keys_to_do  = list(required_input_keys)  # clone
    output_keys_to_do = list(required_output_keys) # clone

    for set in sets:
        regular_length = get_lenght_of_set(name="regular", set=set) * _config().test_time_augmentations
        num_batches = int(np.ceil(regular_length / float(_config().batch_size)))
        regular_chunk_size = _config().batches_per_chunk * _config().batch_size
        num_chunks = int(np.ceil(num_batches / float(_config().batches_per_chunk)))

        indices_for_this_set = xrange(regular_length)
        indices_for_this_set = list(itertools.chain.from_iterable(
            itertools.repeat(x, _config().test_time_augmentations) for x in indices_for_this_set))

        for n in xrange(num_chunks):

            test_sample_numbers = range(n*regular_chunk_size, (n+1)*regular_chunk_size)
            indices = [indices_for_this_set[i] for i in test_sample_numbers if i<len(indices_for_this_set)]
            kaggle_data = get_patient_data(indices, input_keys_to_do, output_keys_to_do,
                                           set=set,
                                           preprocess_function=_config().preprocess_test,
                                           testaug=True)

            yield kaggle_data


def get_number_of_validation_samples(set="validation"):

    sunny_length = get_lenght_of_set(name="sunny", set=set)
    regular_length = get_lenght_of_set(name="regular", set=set)
    return min(sunny_length, regular_length)


def get_number_of_test_batches(sets=["train", "validation", "test"]):
    num_batches = 0
    for set in sets:
        regular_length = get_lenght_of_set(name="regular", set=set) * _config().test_time_augmentations
        num_batches += int(np.ceil(regular_length / float(_config().batch_size)))

    return num_batches



def get_lenght_of_set(name="sunny", set="validation"):
    if name == "sunny":
        if set=="train":
            return len(sunny_train_images)
        elif set=="validation":
            return len(sunny_validation_images)
    else:
        if set=="train":
            return len(train_patient_folders)
        elif set=="validation":
            return len(validation_patient_folders)
        elif set=="test":
            return len(test_patient_folders)


def get_slice_ids_for_patient(id):
    set, index = id_to_index_map[id]
    return [
        _extract_slice_id_from_path(f)
        for f in _in_folder(patient_folders[set][index]) if 'sax' in f]
