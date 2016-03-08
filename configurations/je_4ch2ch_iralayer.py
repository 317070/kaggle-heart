"""Single slice vgg with normalised scale.
"""
import functools

import lasagne as nn
import numpy as np
import theano
import theano.tensor as T

import data_loader
import deep_learning_layers
import image_transform
import layers
import preprocess
import postprocess
import objectives
import theano_printer
import updates
import utils

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

# Training (schedule) parameters
# - batch sizes
batch_size = 16
sunny_batch_size = 4
batches_per_chunk = 16
AV_SLICE_PER_PAT = 11
num_epochs_train = 100

# - learning rate and method
base_lr = .0001
learning_rate_schedule = {
    0: base_lr,
    num_epochs_train*8/10: base_lr/10,
}
momentum = 0.9
build_updates = updates.build_adam_updates

# Preprocessing stuff
cleaning_processes = []
cleaning_processes_post = [functools.partial(preprocess.normalize_contrast_zmuv, z=2)]

preprocess_train = functools.partial(preprocess.preprocess_normscale)
preprocess_validation = functools.partial(preprocess_train, augment=False)
preprocess_test = preprocess_train


augmentation_params = {
    "rotate": (0, 0),
    "shear": (0, 0),
    "zoom_x": (1.0, 1.0),
    "zoom_y": (1.0, 1.0),
    "skew_x": (0, 0),
    "skew_y": (0, 0),
    "translate": (0, 0),
    "flip_vert": (0, 0),
    "roll_time": (0, 0),
    "flip_time": (0, 0)
}

def filter_samples(folders):
    # don't use patients who don't have more than 6 slices
    return [
        folder for folder in folders
        if data_loader.compute_nr_slices(folder) > 6]


sunny_preprocess_train = preprocess.sunny_preprocess_with_augmentation
sunny_preprocess_validation = preprocess.sunny_preprocess_validation
sunny_preprocess_test = preprocess.sunny_preprocess_validation

# Data generators
create_train_gen = data_loader.generate_train_batch
create_eval_valid_gen = functools.partial(data_loader.generate_validation_batch, set="validation")
create_eval_train_gen = functools.partial(data_loader.generate_validation_batch, set="train")
create_test_gen = functools.partial(data_loader.generate_test_batch, set=["validation", "test"])

# Input sizes
image_size = 64
nr_frames = 30
data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, nr_frames, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:chanzoom:4ch": (batch_size, nr_frames, image_size, image_size), # 30 time steps, 100 px wide, 100 px high,
    "sliced:data:chanzoom:2ch": (batch_size, nr_frames, image_size, image_size), # 30 time steps, 100 px wide, 100 px high,
    "sliced:data:ax": (batch_size, nr_frames, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
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
test_time_augmentations = 20 * AV_SLICE_PER_PAT  # More augmentations since a we only use single slices


def lb_softplus(lb):
    return lambda x: nn.nonlinearities.softplus(x) + lb


# Architecture
def build_model(input_layer = None):

    #################
    # Regular model #
    #################

    l_4ch = nn.layers.InputLayer(data_sizes["sliced:data:chanzoom:4ch"])
    l_2ch = nn.layers.InputLayer(data_sizes["sliced:data:chanzoom:2ch"])

    # Add an axis to concatenate over later
    l_4chr = nn.layers.ReshapeLayer(l_4ch, (-1, 1, ) + l_4ch.output_shape[1:])
    l_2chr = nn.layers.ReshapeLayer(l_2ch, (-1, 1, ) + l_2ch.output_shape[1:])
    
    # Cut the images in half, flip the left ones
    l_4ch_left = nn.layers.SliceLayer(l_4ch, indices=slice(image_size//2-1, None, -1), axis=-1)
    l_4ch_right = nn.layers.SliceLayer(l_4ch, indices=slice(image_size//2, None, 1), axis=-1)
    l_2ch_left = nn.layers.SliceLayer(l_2ch, indices=slice(image_size//2-1, None, -1), axis=-1)
    l_2ch_right = nn.layers.SliceLayer(l_2ch, indices=slice(image_size//2, None, 1), axis=-1)

    # Concatenate over second axis
    l_24lr = nn.layers.ConcatLayer([l_4ch_left, l_4ch_right, l_2ch_left, l_2ch_right], axis=1)

    # Move second axis to batch, process them all in the same way
    l_halves = nn.layers.ReshapeLayer(l_24lr, (-1, nr_frames, image_size, image_size//2))

    # First, do some convolutions in all directions
    l1a = nn.layers.dnn.Conv2DDNNLayer(l_halves,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1b = nn.layers.dnn.Conv2DDNNLayer(l1a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1 = nn.layers.dnn.MaxPool2DDNNLayer(l1b, pool_size=(1,2), stride=(1,2))

    # Then, only use the last axis
    l2a = nn.layers.dnn.Conv2DDNNLayer(l1,  W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2b = nn.layers.dnn.Conv2DDNNLayer(l2a, W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2 = nn.layers.dnn.MaxPool2DDNNLayer(l2b, pool_size=(1,2), stride=(1,2))

    l3a = nn.layers.dnn.Conv2DDNNLayer(l2,  W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3b = nn.layers.dnn.Conv2DDNNLayer(l3a, W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3c = nn.layers.dnn.Conv2DDNNLayer(l3b, W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3 = nn.layers.dnn.MaxPool2DDNNLayer(l3c, pool_size=(1,2), stride=(1,2))

    l4a = nn.layers.dnn.Conv2DDNNLayer(l3,  W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4b = nn.layers.dnn.Conv2DDNNLayer(l4a, W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4c = nn.layers.dnn.Conv2DDNNLayer(l4b, W=nn.init.Orthogonal("relu"), filter_size=(1,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4 = nn.layers.dnn.MaxPool2DDNNLayer(l4c, pool_size=(1,2), stride=(1,2))

    # Now, process each row seperately, by flipping the channel and height axis, and then putting height in the batch
    l4shuffle = nn.layers.DimshuffleLayer(l4, pattern=(0,2,1,3))
    l4rows = nn.layers.ReshapeLayer(l4shuffle, (-1, l4shuffle.output_shape[-2], l4shuffle.output_shape[-1]))

    # Systole
    ldsys1 = nn.layers.DenseLayer(l4rows, num_units=256, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys1drop = nn.layers.dropout(ldsys1, p=0.5)
    ldsys2 = nn.layers.DenseLayer(ldsys1drop, num_units=256, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys2drop = nn.layers.dropout(ldsys2, p=0.5)
    ldsys3mu = nn.layers.DenseLayer(ldsys2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(16.0), nonlinearity=None)
    ldsys3sigma = nn.layers.DenseLayer(ldsys2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(4.0), nonlinearity=lb_softplus(.01))
    ldsys3musigma = nn.layers.ConcatLayer([ldsys3mu, ldsys3sigma], axis=1)

    l_24lr_sys_musigma = nn.layers.ReshapeLayer(ldsys3musigma, (-1, 4, image_size, 2))
    l_4ch_left_sys_musigma = nn.layers.SliceLayer(l_24lr_sys_musigma, indices=0, axis=1)   
    l_4ch_right_sys_musigma = nn.layers.SliceLayer(l_24lr_sys_musigma, indices=1, axis=1)   
    l_2ch_left_sys_musigma = nn.layers.SliceLayer(l_24lr_sys_musigma, indices=2, axis=1)   
    l_2ch_right_sys_musigma = nn.layers.SliceLayer(l_24lr_sys_musigma, indices=3, axis=1)

    l_4ch_sys_musigma = layers.SumGaussLayer([l_4ch_left_sys_musigma, l_4ch_right_sys_musigma])
    l_2ch_sys_musigma = layers.SumGaussLayer([l_2ch_left_sys_musigma, l_2ch_right_sys_musigma])

    l_sys_musigma = layers.IraLayerNoTime(l_4ch_sys_musigma, l_2ch_sys_musigma)

    l_systole = layers.MuSigmaErfLayer(l_sys_musigma)

    # Systole
    lddia1 = nn.layers.DenseLayer(l4rows, num_units=256, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia1drop = nn.layers.dropout(lddia1, p=0.5)
    lddia2 = nn.layers.DenseLayer(lddia1drop, num_units=256, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia2drop = nn.layers.dropout(lddia2, p=0.5)
    lddia3mu = nn.layers.DenseLayer(lddia2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(16.0), nonlinearity=None)
    lddia3sigma = nn.layers.DenseLayer(lddia2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(4.0), nonlinearity=lb_softplus(.01))
    lddia3musigma = nn.layers.ConcatLayer([lddia3mu, lddia3sigma], axis=1)

    l_24lr_dia_musigma = nn.layers.ReshapeLayer(lddia3musigma, (-1, 4, image_size, 2))
    l_4ch_left_dia_musigma = nn.layers.SliceLayer(l_24lr_dia_musigma, indices=0, axis=1)   
    l_4ch_right_dia_musigma = nn.layers.SliceLayer(l_24lr_dia_musigma, indices=1, axis=1)   
    l_2ch_left_dia_musigma = nn.layers.SliceLayer(l_24lr_dia_musigma, indices=2, axis=1)   
    l_2ch_right_dia_musigma = nn.layers.SliceLayer(l_24lr_dia_musigma, indices=3, axis=1)

    l_4ch_dia_musigma = layers.SumGaussLayer([l_4ch_left_dia_musigma, l_4ch_right_dia_musigma])
    l_2ch_dia_musigma = layers.SumGaussLayer([l_2ch_left_dia_musigma, l_2ch_right_dia_musigma])

    l_dia_musigma = layers.IraLayerNoTime(l_4ch_dia_musigma, l_2ch_dia_musigma)

    l_diastole = layers.MuSigmaErfLayer(l_dia_musigma)

    return {
        "inputs":{
            "sliced:data:chanzoom:4ch": l_4ch,
            "sliced:data:chanzoom:2ch": l_2ch,
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": {
        },
        "meta_outputs": {
        }
    }

