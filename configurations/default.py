import numpy as np
from preprocess import preprocess
import lasagne
from updates import build_updates
from data_loader import generate_train_batch, generate_validation_batch
from functools import partial

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

preprocess = preprocess
build_updates = build_updates

batch_size = 128
chunk_size = 32768
num_chunks_train = 840

validate_every = 20
save_every = 20
restart_from_save = True

learning_rate_schedule = {
    0: 0.003,
    400: 0.0003,
    500: 0.00003,
}

