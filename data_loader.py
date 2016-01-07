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
validation_patients_indices = get_cross_validation_indices()

VALIDATION_REGEX = r"/" + "|".join(["%d"%i for i in validation_patients_indices])+"/"

train_patient_folders = [folder for folder in patient_folders if re.match(VALIDATION_REGEX, folder) is None]
validation_patient_folders = [folder for folder in patient_folders if folder not in train_patient_folders]

regular_labels = pickle.load(open("/data/dsb15_pkl/train.pkl","r"))

##############
# Sunny data #
##############

sunny_data = pickle.load(open("/data/dsb15_pkl/pkl_annotated/data.pkl","rb"))
num_sunny_images = len(sunny_data["images"])

validation_sunny_indices = np.random.choice(range(num_sunny_images), 50, replace=False)
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
            "systole": np.zeros((config().chunk_size, ), dtype='float32'),
            "diastole": np.zeros((config().chunk_size, ), dtype='float32'),
        }
    }

    for tag in wanted_data_tags:
        if tag in config().data_sizes:
            result["input"][tag] = np.zeros((config().chunk_size, ) + config().data_sizes[tag][1:], dtype='float32')

    if set=="train":
        folders = [train_patient_folders[i] for i in indices]
    else:
        folders = [validation_patient_folders[i] for i in indices]

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
        result["output"]["systole"][i] = regular_labels[id-1, 1]
        result["output"]["diastole"][i] = regular_labels[id-1, 2]
    return result


def get_sunny_patient_data(indices, set="train"):
    images = sunny_train_images
    labels = sunny_train_labels

    sunny_chunk = np.zeros((config().chunk_size, 1, 256, 256), dtype='float32')
    sunny_label_chunk = np.zeros((config().chunk_size, 256, 256),    dtype='float32')

    for k, idx in enumerate(indices):
        img = images[indices[k]]-128
        lbl = labels[indices[k]]
        config().sunny_preprocess(sunny_chunk[k], img, sunny_label_chunk[k], lbl)

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

    for n in xrange(config().num_chunks_train):
        result = {}
        input_keys_to_do = list(required_input_keys) #clone
        output_keys_to_do = list(required_output_keys) #clone
        if "sunny" in input_keys_to_do or "segmentation" in output_keys_to_do:
            indices = config().rng.randint(0, len(sunny_train_images), config().sunny_chunk_size)
            sunny_patient_data = get_sunny_patient_data(indices, set="train")
            result = utils.merge(result, sunny_patient_data)
            input_keys_to_do.remove("sunny")
            output_keys_to_do.remove("segmentation")

        indices = config().rng.randint(0, len(train_patient_folders), config().chunk_size)
        kaggle_data = get_patient_data(indices, input_keys_to_do, set="train")

        result = utils.merge(result, kaggle_data)

        yield result


# TODO: finish this function

def generate_validation_batch(required_input_keys, required_output_keys):
    # generate sunny data
    if set=="sunny:train":
        images = sunny_train_images
        labels = sunny_train_labels
    elif set=="sunny:validation":
        images = sunny_validation_images
        labels = sunny_validation_labels
    else:
        raise "choose either validation or train set"

    num_images = len(images)
    num_chunks = int(np.ceil(num_images / float(config().chunk_size)))

    idx = 0
    for n in xrange(num_chunks):

        current_chunk_length = config().chunk_size
        if n*config().chunk_size + current_chunk_length > len(num_images):
            current_chunk_length = len(num_images) - n*config().chunk_size

        indices = range(n*config().chunk_size, n*config().chunk_size + current_chunk_length)

        result = {}
        input_keys_to_do = list(required_input_keys) #clone
        output_keys_to_do = list(required_output_keys) #clone

        if "sunny" in input_keys_to_do or "segmentation" in output_keys_to_do:
            sunny_patient_data = get_sunny_patient_data(indices, set="train")
            result = utils.merge(result, sunny_patient_data)
            input_keys_to_do.remove("sunny")
            output_keys_to_do.remove("segmentation")

        kaggle_data = get_patient_data(indices, input_keys_to_do, set="train")

        result = utils.merge(result, kaggle_data)
        idx += 1
        yield result, current_chunk_length


def generate_validation_batch(set="validation"):
    if set=="train":
        images = sunny_train_images
        labels = sunny_train_labels
    elif set=="validation":
        images = sunny_validation_images
        labels = sunny_validation_labels
    else:
        raise "choose either validation or train set"

    num_images = len(images)
    num_chunks = int(np.ceil(num_images / float(config().chunk_size)))

    idx = 0

    for n in xrange(num_chunks):
        chunk_size = config().chunk_size
        chunk_x = np.zeros((chunk_size, 1, 256, 256), dtype='float32')
        chunk_labels = np.zeros((chunk_size, 256, 256), dtype='float32')
        current_chunk_length = chunk_size

        for k in xrange(chunk_size):
            if idx >= num_images:
                current_chunk_length = k
                break

            img = images[idx]
            lbl = labels[idx]
            config().preprocess_validation(chunk_x[k], img, chunk_labels[k], lbl)
            idx += 1

        yield [chunk_x], current_chunk_length


def get_label_set(set="validation"):
    if set=="train":
        images = sunny_train_images
        labels = sunny_train_labels
    elif set=="validation":
        images = sunny_validation_images
        labels = sunny_validation_labels
    else:
        raise "choose either validation or train set"
    return images, labels


