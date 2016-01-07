import numpy as np
from preprocess import sunny_preprocess, sunny_preprocess_validation, preprocess
import lasagne
from updates import build_updates
from data_loader import generate_train_batch, generate_validation_batch, get_label_set
from functools import partial
import lasagne
import lasagne.layers.cuda_convnet

"""
When adding new configuration parameters, add the default values to this config file. This adds them to
all old config files automatically. Make sure this parameter does not change
the algorithm of the old files.
"""

momentum = 0.9
rng = np.random

create_train_gen = generate_train_batch
create_eval_valid_gen = partial(generate_validation_batch, set="validation")
create_eval_train_gen = partial(generate_validation_batch, set="train")

sunny_preprocess = sunny_preprocess
sunny_preprocess_validation = sunny_preprocess_validation
preprocess = preprocess

build_updates = build_updates

get_label_set = get_label_set

# In total, you train 'chunk_size' samples 'num_chunks_train' time, and you do updates every 'batch_size'
sunny_batch_size = 128
sunny_chunk_size = 4096
batch_size = 128
chunk_size = 4096
num_chunks_train = 840

validate_every = 20
save_every = 20
restart_from_save = True

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