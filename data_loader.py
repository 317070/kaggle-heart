import glob
import numpy as np
import re
from configuration import config
import cPickle as pickle
import utils
from validation_set import get_cross_validation_indices
import random
from itertools import chain

print "Loading data"
################
# Regular data #
################

patient_folders = sorted(glob.glob("/data/dsb15_pkl/pkl_train/*/study/"), key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))  # glob is non-deterministic!

validation_patients_indices = get_cross_validation_indices(indices=range(1,501), validation_index=0)
train_patients_indices = [i for i in range(1,501) if i not in validation_patients_indices]

VALIDATION_REGEX = "|".join(["(/%d/)"%i for i in validation_patients_indices])

train_patient_folders = [folder for folder in patient_folders if re.search(VALIDATION_REGEX, folder) is None]
validation_patient_folders = [folder for folder in patient_folders if folder not in train_patient_folders]

regular_labels = pickle.load(open("/data/dsb15_pkl/train.pkl","r"))

test_patient_folders = sorted(glob.glob("/data/dsb15_pkl/pkl_validate/*/study/"), key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))  # glob is non-deterministic!


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


def get_patient_data(indices, wanted_input_tags, wanted_output_tags, set="train",
                     train_preprocess=False, validate_preprocess=False, test_preprocess=False):
    """
    return a dict with the desired data matched to the required tags
    :param wanted_data_tags:
    :return:
    """
    result = {
        "input": {
        },
        "output": {
        },
    }

    for tag in wanted_output_tags:
        if tag in ["systole", "diastole", "systole:onehot", "diastole:onehot", "systole:class_weight", "diastole:class_weight"]:
            result["output"][tag] = np.zeros((config().batch_size * config().batches_per_chunk, 600), dtype='float32')
        if tag in ["systole:value", "diastole:value"]:
            result["output"][tag] = np.zeros((config().batch_size * config().batches_per_chunk, ), dtype='float32')
        # and for the predictions, keep track of which patient is sitting where in the batch
        if tag=="patients":
            result["output"][tag] = np.zeros((config().batch_size * config().batches_per_chunk, ), dtype='int32')

    for tag in wanted_input_tags:
        if tag in config().data_sizes:
            chunk_shape = list(config().data_sizes[tag])
            chunk_shape[0] = chunk_shape[0] * config().batches_per_chunk
            chunk_shape = tuple(chunk_shape)
            result["input"][tag] = np.zeros(chunk_shape, dtype='float32')

    if set=="train":
        folders = [train_patient_folders[i] for i in indices if 0<=i<len(train_patient_folders)]
    elif set=="validation":
        folders = [validation_patient_folders[i] for i in indices if 0<=i<len(validation_patient_folders)]
    elif set=="test":
        folders = [test_patient_folders[i] for i in indices if 0<=i<len(test_patient_folders)]
    else:
        raise "Don't know the dataset %s" % set


    for i, folder in enumerate(folders):
        files = sorted(glob.glob(folder+"*"))  # glob is non-deterministic!
        patient_result = dict()
        for tag in wanted_input_tags:
            if tag.startswith("sliced:data:singleslice"):
                l = [sax for sax in files if "sax" in sax]
                if "middle" in tag:
                    f = l[len(l)/2]
                else:
                    f = random.choice(l)
                patient_result[tag] = pickle.load(open(f, "r"))['data'].astype('float32')
                if "difference" in tag:
                    for j in xrange(patient_result[tag].shape[0]-1):
                        patient_result[tag][j] -= patient_result[tag][j+1]
                    patient_result[tag] = np.delete(patient_result[tag],-1,0)
            elif tag.startswith("sliced:data"):
                patient_result[tag] = [pickle.load(open(f, "r"))['data'].astype('float32') for f in files]
            elif tag.startswith("sliced:data:ax"):
                patient_result[tag] = [pickle.load(open(f, "r"))['data'].astype('float32') for f in files if "sax" in f]
            elif tag.startswith("sliced:data:shape"):
                patient_result[tag] = [pickle.load(open(f, "r"))['data'].shape for f in files]
            elif tag.startswith("sliced:meta:"):
                # get the key used in the pickle
                key = tag[len("slided:meta:"):]
                patient_result[tag] = [pickle.load(open(f, "r"))['metadata'][key] for f in files]
            # add others when needed

        if train_preprocess:
            config().preprocess_train(patient_result, result=result["input"], index=i)
        elif validate_preprocess:
            config().preprocess_validation(patient_result, result=result["input"], index=i)
        elif test_preprocess:
            config().preprocess_test(patient_result, result=result["input"], index=i)


        # load the labels
        # find the id of the current patient in the folder name (=safer)
        id = int(re.search(r'/(\d+)/', folder).group(1))
        if "patients" in wanted_output_tags:
            result["output"]["patients"][i] = id

        # only read labels, when we actually have them
        if id in regular_labels[:, 0]:
            assert regular_labels[id-1, 0]==id
            V_systole = regular_labels[id-1, 1]
            V_diastole = regular_labels[id-1, 2]
            if "systole:onehot" in wanted_output_tags:
                result["output"]["systole:onehot"][i][int(np.ceil(V_systole))] = 1
            if "diastole:onehot" in wanted_output_tags:
                result["output"]["diastole:onehot"][i][int(np.ceil(V_diastole))] = 1
            if "systole" in wanted_output_tags:
                result["output"]["systole"][i][int(np.ceil(V_systole)):] = 1
            if "diastole" in wanted_output_tags:
                result["output"]["diastole"][i][int(np.ceil(V_diastole)):] = 1
            if "systole:value" in wanted_output_tags:
                result["output"]["systole:value"][i] = V_systole
            if "diastole:value" in wanted_output_tags:
                result["output"]["diastole:value"][i] = V_diastole
            if "diastole:class_weight" in wanted_output_tags:
                result["output"]["diastole:class_weight"][i] = utils.linear_weighted(V_diastole)
            if "systole:class_weight" in wanted_output_tags:
                result["output"]["systole:class_weight"][i] = utils.linear_weighted(V_systole)


    # Check if any of the inputs or outputs are still empty!
    for key, value in chain(result["input"].iteritems(), result["output"].iteritems()):
        if not np.any(value): #there are only zeros in value
            raise Exception("there is an empty value at key %s" % key)
        if not np.isfinite(value).all(): #there are NaN's or infinites somewhere
            print value
            raise Exception("there is a NaN at key %s" % key)

        if set=="train" and sum([0 if 0<=i<len(train_patient_folders) else 1 for i in indices]) > 0:
            raise Exception("not filled train batch at key %s" % key)

        for idx, sample in enumerate(value[:len(folders)]):
            if not np.any(value): #there are only zeros in value
                print value
                raise Exception("there is an empty sample at key %s" % key)

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

        indices = config().rng.randint(0, len(train_patient_folders), chunk_size)  #
        kaggle_data = get_patient_data(indices, input_keys_to_do, output_keys_to_do, set="train",
                                       train_preprocess=True)

        result = utils.merge(result, kaggle_data)

        yield result


def generate_validation_batch(required_input_keys, required_output_keys, set="validation"):
    # generate sunny data
    sunny_length = get_lenght_of_set(name="sunny", set=set)
    regular_length = get_lenght_of_set(name="regular", set=set)

    sunny_batches = int(np.ceil(sunny_length / float(config().sunny_batch_size)))
    regular_batches = int(np.ceil(regular_length / float(config().batch_size)))

    if "sunny" in required_input_keys or "segmentation" in required_output_keys:
        num_batches = max(sunny_batches, regular_batches)
    else:
        num_batches = regular_batches

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
        kaggle_data = get_patient_data(indices, input_keys_to_do, output_keys_to_do, set=set,
                                       validate_preprocess=True)

        result = utils.merge(result, kaggle_data)

        yield result


def generate_test_batch(required_input_keys, required_output_keys, set="test"):
    regular_length = get_lenght_of_set(name="regular", set=set)
    num_batches = int(np.ceil(regular_length / float(config().batch_size)))
    regular_chunk_size = config().batches_per_chunk * config().batch_size
    num_chunks = int(np.ceil(num_batches / float(config().batches_per_chunk)))

    for n in xrange(num_chunks):

        input_keys_to_do  = list(required_input_keys)  # clone
        output_keys_to_do = list(required_output_keys) # clone

        indices = range(n*regular_chunk_size, (n+1)*regular_chunk_size)
        kaggle_data = get_patient_data(indices, input_keys_to_do, output_keys_to_do, set=set,
                                       test_preprocess=True)

        yield kaggle_data


def get_number_of_validation_samples(set="validation"):

    sunny_length = get_lenght_of_set(name="sunny", set=set)
    regular_length = get_lenght_of_set(name="regular", set=set)
    return min(sunny_length, regular_length)


def get_number_of_test_batches(set="test"):
    regular_length = get_lenght_of_set(name="regular", set=set)
    num_batches = int(np.ceil(regular_length / float(config().batch_size)))

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

