import numpy as np
import objectives
from preprocess import sunny_preprocess, sunny_preprocess_validation, preprocess, preprocess_with_augmentation, \
    sunny_preprocess_with_augmentation
import lasagne
from updates import build_nesterov_updates
from data_loader import generate_train_batch, generate_validation_batch, generate_test_batch
from functools import partial
import lasagne
from postprocess import postprocess

"""
When adding new configuration parameters, add the default values to this config file. This adds them to
all old config files automatically. Make sure this parameter does not change
the algorithm of the old files.
"""

caching = None  # "memory"  # by default, cache accessed files in memory
momentum = 0.9
rng = np.random


create_train_gen = generate_train_batch
create_eval_valid_gen = partial(generate_validation_batch, set="validation")
create_eval_train_gen = partial(generate_validation_batch, set="train")
create_test_gen = partial(generate_test_batch, set=None)  # validate as well by default

sunny_preprocess_train = sunny_preprocess_with_augmentation
sunny_preprocess_validation = sunny_preprocess_validation
sunny_preprocess_test = sunny_preprocess_validation

preprocess_train = preprocess_with_augmentation
preprocess_validation = preprocess
preprocess_test = preprocess

cleaning_processes = []

test_time_augmentations = 100

postprocess = postprocess

build_updates = build_nesterov_updates

# In total, you train 'chunk_size' samples 'num_chunks_train' time, and you do updates every 'batch_size'
# you train until the train set has passed by 'num_epochs_train' times
num_epochs_train = 150

validate_every = 20
save_every = 20
restart_from_save = False
take_a_dump = False  # dump a lot of data in a pkl-dump file. (for debugging)
dump_network_loaded_data = False  # dump the outputs from the dataloader (for debugging)

augmentation_params = {
    "rotation": (0, 360),
    "shear": (-10, 10),
    "translation": (-8, 8),
}

data_sizes = {
    "sliced:data": (30, 30, 100, 100), #30 mri_slices, 30 time steps, 100 px wide, 100 px high,
    "sliced:data:shape": (2,),
    # TBC with the metadata
}

learning_rate_schedule = {
    0: 0.003,
    400: 0.0003,
    500: 0.00003,
}

def build_model():
    return {
        "inputs":[],
        "output":None
    }

def build_objective(l_ins, l_outs):
    return objectives.LogLossObjective(l_outs)