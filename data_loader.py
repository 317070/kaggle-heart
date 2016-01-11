import glob
import numpy as np
import re
from configuration import config
import cPickle as pickle
import utils
from validation_set import get_cross_validation_indices


################
# Regular data #
################

patient_folders = sorted(glob.glob("/data/dsb15_pkl/pkl_train/*/study/"))  # glob is non-deterministic!
validation_patients_indices = get_cross_validation_indices(indices=range(1,501))
train_patients_indices = [i for i in range(1,501) if i not in validation_patients_indices]

VALIDATION_REGEX = "|".join(["(/%d/)"%i for i in validation_patients_indices])

train_patient_folders = [folder for folder in patient_folders if re.search(VALIDATION_REGEX, folder) is None]
validation_patient_folders = [folder for folder in patient_folders if folder not in train_patient_folders]

regular_labels = pickle.load(open("/data/dsb15_pkl/train.pkl","r"))


##############
# Sunny data #
##############

sunny_data = pickle.load(open("/data/dsb15_pkl/pkl_annotated/data.pkl","rb"))
num_sunny_images = len(sunny_data["images"])

validation_sunny_indices = get_cross_validation_indices(indices=range(num_sunny_images))
train_sunny_indices = [i for i in range(num_sunny_images) if i not in validation_sunny_indices]

sunny_train_images = np.array(sunny_data['images'])[train_sunny_indices]
sunny_train_labels = np.array(sunny_data['labels'])[train_sunny_indices]

sunny_validation_images = np.array(sunny_data['images'])[validation_sunny_indices]
sunny_validation_labels = np.array(sunny_data['labels'])[validation_sunny_indices]


def get_patient_data(indices, wanted_data_tags, set="train"):
    """
    return a dict with the desired data matched to the required tags
    :param wanted_data_tags:
    :return:
    """
    result = {
        "input": {
        },
        "output": {
            "systole":  np.zeros((config().batch_size * config().batches_per_chunk, 600), dtype='float32'),
            "diastole": np.zeros((config().batch_size * config().batches_per_chunk, 600), dtype='float32'),
        }
    }

    for tag in wanted_data_tags:
        if tag in config().data_sizes:
            chunk_shape = list(config().data_sizes[tag])
            chunk_shape[0] = chunk_shape[0] * config().batches_per_chunk
            chunk_shape = tuple(chunk_shape)
            result["input"][tag] = np.zeros(chunk_shape, dtype='float32')

    if set=="train":
        folders = [train_patient_folders[i] for i in indices if i<len(train_patient_folders)]
    else:
        folders = [validation_patient_folders[i] for i in indices if i<len(train_patient_folders)]

    # TODO: every folder is multiple times in a chunk!
    # therefore, this can be done with loading every file only once

    for i, folder in enumerate(folders):
        files = sorted(glob.glob(folder+"*"))  # glob is non-deterministic!
        patient_result = dict()
        for tag in wanted_data_tags:
            if tag.startswith("sliced:data"):
                patient_result[tag] = [pickle.load(open(f, "r"))['data'] for f in files]
            if tag.startswith("sliced:data:shape"):
                patient_result[tag] = [pickle.load(open(f, "r"))['data'].shape for f in files]
            if tag.startswith("sliced:meta:"):
                # get the key used in the pickle
                key = tag[len("slided:meta:"):]
                patient_result[tag] = [pickle.load(open(f, "r"))['metadata'][key] for f in files]
            # add others when needed
        config().preprocess(patient_result, result=result["input"], index=i)

        # load the labels
        # find the id of the current patient in the folder name (=safer)
        id = int(re.search(r'/(\d+)/', folder).group(1))
        assert regular_labels[id-1, 0]==id
        V_systole = regular_labels[id-1, 1]
        V_diastole = regular_labels[id-1, 2]

        result["output"]["systole"][i][int(np.ceil(V_systole)):] = 1
        result["output"]["diastole"][i][int(np.ceil(V_diastole)):] = 1
    return result


def get_sunny_patient_data(indices, set="train"):
    images = sunny_train_images
    labels = sunny_train_labels

    chunk_shape = list(config().data_sizes["sunny"])
    chunk_shape[0] = chunk_shape[0] * config().batches_per_chunk
    chunk_shape = tuple(chunk_shape)

    sunny_chunk = np.zeros(chunk_shape, dtype='float32')

    chunk_shape = list(config().data_sizes["sunny"])
    chunk_shape[0] = chunk_shape[0] * config().batches_per_chunk
    chunk_shape.remove(1)
    chunk_shape = tuple(chunk_shape)

    sunny_label_chunk = np.zeros(chunk_shape, dtype='float32')

    for k, idx in enumerate(indices):
        if indices[k]<len(images):
            img = images[indices[k]]
            lbl = labels[indices[k]]
            config().sunny_preprocess(sunny_chunk[k], img, sunny_label_chunk[k], lbl)
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


def generate_train_batch(required_input_keys, required_output_keys):
    # generate sunny data

    sunny_chunk_size = config().sunny_batch_size * config().batches_per_chunk
    chunk_size = config().batch_size * config().batches_per_chunk

    for n in xrange(config().num_chunks_train):
        result = {}
        input_keys_to_do = list(required_input_keys) #clone
        output_keys_to_do = list(required_output_keys) #clone
        if "sunny" in input_keys_to_do or "segmentation" in output_keys_to_do:
            indices = config().rng.randint(0, len(sunny_train_images), sunny_chunk_size)
            sunny_patient_data = get_sunny_patient_data(indices, set="train")
            result = utils.merge(result, sunny_patient_data)
            input_keys_to_do.remove("sunny")
            output_keys_to_do.remove("segmentation")

        indices = config().rng.randint(0, len(train_patient_folders), chunk_size)
        kaggle_data = get_patient_data(indices, input_keys_to_do, set="train")

        result = utils.merge(result, kaggle_data)

        yield result


def generate_validation_batch(required_input_keys, required_output_keys, set="train"):
    # generate sunny data
    sunny_length = get_lenght_of_set(name="sunny", set=set)
    regular_length = get_lenght_of_set(name="regular", set=set)

    sunny_batches = int(np.ceil(sunny_length / float(config().sunny_batch_size)))
    regular_batches = int(np.ceil(regular_length / float(config().batch_size)))

    num_batches = max(sunny_batches, regular_batches)

    num_chunks = int(np.ceil(num_batches / float(config().batches_per_chunk)))

    sunny_chunk_size = config().batches_per_chunk * config().sunny_batch_size
    regular_chunk_size = config().batches_per_chunk * config().batch_size

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
        kaggle_data = get_patient_data(indices, input_keys_to_do, set="train")

        result = utils.merge(result, kaggle_data)

        yield result


def get_number_of_validation_batches(set="validation"):

    sunny_length = get_lenght_of_set(name="sunny", set=set)
    regular_length = get_lenght_of_set(name="regular", set=set)

    sunny_batches = int(np.ceil(sunny_length / float(config().sunny_batch_size)))
    regular_batches = int(np.ceil(regular_length / float(config().batch_size)))
    return min(sunny_batches, regular_batches)


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

