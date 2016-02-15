"""Single slice vgg with normalised scale.
"""
import functools

import lasagne as nn
import numpy as np
import theano
import theano.tensor as T

import data_loader
import deep_learning_layers
import layers
import preprocess
import postprocess
import objectives
import theano_printer
import updates

# Random params
rng = np.random
take_a_dump = False  # dump a lot of data in a pkl-dump file. (for debugging)
dump_network_loaded_data = False  # dump the outputs from the dataloader (for debugging)

# Memory usage scheme
caching = None

# Save and validation frequency
validate_every = 10
validate_train_set = True
save_every = 10
restart_from_save = False

dump_network_loaded_data = False

# Training (schedule) parameters
# - batch sizes
batch_size = 32
sunny_batch_size = 4
batches_per_chunk = 16
AV_SLICE_PER_PAT = 11
num_epochs_train = 100 * AV_SLICE_PER_PAT

# - learning rate and method
base_lr = 0.0003
learning_rate_schedule = {
    0: 10*base_lr,
    21: base_lr,
    4*num_epochs_train/5: base_lr/10, 
}
momentum = 0.9
build_updates = updates.build_adam_updates

# Preprocessing stuff
cleaning_processes = [
    preprocess.set_upside_up,]
cleaning_processes_post = [
    functools.partial(preprocess.normalize_contrast_zmuv, z=2)]

augmentation_params = {
    "rotation": (-16, 16),
    "shear": (0, 0),
    "translation": (-8, 8),
    "flip_vert": (0, 1)
}

preprocess_train = preprocess.preprocess_normscale
preprocess_validation = functools.partial(preprocess_train, augment=False)
preprocess_test = preprocess_train

sunny_preprocess_train = preprocess.sunny_preprocess_with_augmentation
sunny_preprocess_validation = preprocess.sunny_preprocess_validation
sunny_preprocess_test = preprocess.sunny_preprocess_validation

# Data generators
create_train_gen = data_loader.generate_train_batch
create_eval_valid_gen = functools.partial(data_loader.generate_validation_batch, set="validation")
create_eval_train_gen = functools.partial(data_loader.generate_validation_batch, set="train")
create_test_gen = functools.partial(data_loader.generate_test_batch, set=None)  # Eval all sets 

# Input sizes
image_size = 128
data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}

# Objective
l2_weight = 0.000
l2_weight_out = 0.000
def build_objective(interface_layers):
    # l2 regu on certain layers
    l2_penalty = nn.regularization.regularize_layer_params_weighted(
        interface_layers["regularizable"], nn.regularization.l2)
    # build objective
    return objectives.KaggleObjective(interface_layers["outputs"], penalty=l2_penalty)

# Testing
postprocess = postprocess.postprocess
test_time_augmentations = 100 * AV_SLICE_PER_PAT  # More augmentations since a we only use single slices

# Architecture
def build_model():

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:singleslice"]

    l0 = nn.layers.InputLayer(input_size)

    l1a = nn.layers.dnn.Conv2DDNNLayer(l0 , filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1b = nn.layers.dnn.Conv2DDNNLayer(l1a, filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1 = nn.layers.dnn.MaxPool2DDNNLayer(l1b, pool_size=(2,2), stride=(2,2))

    l2a = nn.layers.dnn.Conv2DDNNLayer(l1 , filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2b = nn.layers.dnn.Conv2DDNNLayer(l2a, filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2 = nn.layers.dnn.MaxPool2DDNNLayer(l2b, pool_size=(2,2), stride=(2,2))

    l3a = nn.layers.dnn.Conv2DDNNLayer(l2 , filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3b = nn.layers.dnn.Conv2DDNNLayer(l3a, filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3c = nn.layers.dnn.Conv2DDNNLayer(l3b, filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3 = nn.layers.dnn.MaxPool2DDNNLayer(l3c, pool_size=(2,2), stride=(2,2))

    l4a = nn.layers.dnn.Conv2DDNNLayer(l3 , filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4b = nn.layers.dnn.Conv2DDNNLayer(l4a, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4c = nn.layers.dnn.Conv2DDNNLayer(l4b, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4 = nn.layers.dnn.MaxPool2DDNNLayer(l4c, pool_size=(2,2), stride=(2,2))

    l5a = nn.layers.dnn.Conv2DDNNLayer(l4 , filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5b = nn.layers.dnn.Conv2DDNNLayer(l5a, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5c = nn.layers.dnn.Conv2DDNNLayer(l5b, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5 = nn.layers.dnn.MaxPool2DDNNLayer(l5c, pool_size=(2,2), stride=(2,2))

    # Systole Dense layers
    ldsys1 = nn.layers.DenseLayer(l5, num_units=1024, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys1drop = nn.layers.dropout(ldsys1, p=0.5)
    ldsys2 = nn.layers.DenseLayer(ldsys1drop, num_units=1024, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys2drop = nn.layers.dropout(ldsys2, p=0.5)
    ldsys3 = nn.layers.DenseLayer(ldsys2drop, num_units=600, b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)

    l_systole = layers.CumSumLayer(ldsys3)

    # Diastole Dense layers
    lddia1 = nn.layers.DenseLayer(l5, num_units=1024, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia1drop = nn.layers.dropout(lddia1, p=0.5)
    lddia2 = nn.layers.DenseLayer(lddia1drop, num_units=1024, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia2drop = nn.layers.dropout(lddia2, p=0.5)
    lddia3 = nn.layers.DenseLayer(lddia2drop, num_units=600, b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)

    l_diastole = layers.CumSumLayer(lddia3)


    return {
        "inputs":{
            "sliced:data:singleslice": l0
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": {
            ldsys1: l2_weight,
            ldsys2: l2_weight,
            ldsys3: l2_weight_out,
            lddia1: l2_weight,
            lddia2: l2_weight,
            lddia3: l2_weight_out,
        },
    }

