import glob
import numpy as np
import re
from configuration import config
import cPickle as pickle
from validation_set import get_cross_validation_indices


################
# Regular data #
################

patient_folders = sorted(glob.glob("/data/dsb15_pkl/4d_group_pkl_train/*/study/"))  # glob is non-deterministic!
validation_patients_indices = get_cross_validation_indices()

VALIDATION_REGEX = r"/" + "|".join(["%d"%i for i in validation_patients_indices])+"/"

train_patient_folders = [folder for folder in patient_folders if re.match(VALIDATION_REGEX, folder) is None]
validation_patient_folders = [folder for folder in patient_folders if folder not in train_patient_folders]


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


def get_patient_data(index, wanted_data_tags, set="train"):
    """
    return a dict with the desired data matched to the required tags
    :param wanted_data_tags:
    :return:
    """
    result = dict()
    if set=="train":
        folder = train_patient_folders[index]
    else:
        folder = validation_patient_folders[index]

    files = sorted(glob.glob(folder+"*"))  # glob is non-deterministic!
    for data_tag in wanted_data_tags.keys():
        if data_tag.startswith("sliced:data"):
            result[data_tag] = [pickle.load(open(f, "r"))['data'] for f in files]
        if data_tag.startswith("sliced:meta:"):
            key = data_tag[len("slided:meta:"):]
            result[data_tag] = [pickle.load(open(f, "r"))['metadata'][key] for f in files]
        if data_tag.startswith("sunny"):
            result[data_tag] = [pickle.load(open(f, "r"))['metadata'][key] for f in files]
        # add others when needed
    return result

def generate_train_batch(required_input_keys, required_output_keys):
    images = sunny_train_images
    labels = sunny_train_labels

    for n in xrange(config().num_chunks_train):
        indices = config().rng.randint(0, len(sunny_train_images), config().chunk_size)

        chunk_x = np.zeros((config().chunk_size, 1, 256, 256), dtype='float32')
        chunk_y = np.zeros((config().chunk_size, 256, 256), dtype='float32')

        for k, idx in enumerate(indices):
            img = images[indices[k]]-128
            lbl = labels[indices[k]]
            config().preprocess(chunk_x[k], img, chunk_y[k], lbl)

        yield {
            "input": {
                "sunny": chunk_x,
            },
            "output": {
                "segmentation": chunk_y,
            }
        }


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


