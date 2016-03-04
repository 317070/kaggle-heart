"""Single slice vgg with normalised scale.
"""
import functools

import lasagne as nn
from lasagne.layers import reshape
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
num_epochs_train = 80 * AV_SLICE_PER_PAT

# - learning rate and method
base_lr = .0001
learning_rate_schedule = {
    0: base_lr,
    num_epochs_train*9/10: base_lr/10,
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
data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:chanzoom:4ch": (batch_size, 30, image_size, image_size), # 30 time steps, 100 px wide, 100 px high,
    "sliced:data:chanzoom:2ch": (batch_size, 30, image_size, image_size), # 30 time steps, 100 px wide, 100 px high,
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
test_time_augmentations = 20 * AV_SLICE_PER_PAT  # More augmentations since a we only use single slices
tta_average_method = lambda x: np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))

# Architecture
def build_model(input_layer = None):

    #################
    # Regular model #
    #################

    l_4ch = nn.layers.InputLayer(data_sizes["sliced:data:chanzoom:4ch"])
    l_2ch = nn.layers.InputLayer(data_sizes["sliced:data:chanzoom:2ch"])
    #
    l_4chr = reshape(l_4ch, (-1, 1, ) + l_4ch.output_shape[1:])
    l_2chr = reshape(l_2ch, (-1, 1, ) + l_2ch.output_shape[1:])
    # batch, features, timesteps, width, height

    l_4ch_2a = deep_learning_layers.ConvolutionOver2DAxisLayer(l_4chr, num_filters=16, filter_size=(3, 3),
                                                             axis=(3,4), channel=1,
                                                             pad=(1,1),
                                                             W=nn.init.Orthogonal(),
                                                             b=nn.init.Constant(0.0),
                                                             )
    l_4ch_2b = deep_learning_layers.ConvolutionOver2DAxisLayer(l_4ch_2a, num_filters=16, filter_size=(3, 3),
                                                             axis=(3,4), channel=1,
                                                             pad=(1,1),
                                                             W=nn.init.Orthogonal(),
                                                             b=nn.init.Constant(0.0),
                                                             )
    l_4ch_2m = deep_learning_layers.MaxPoolOverAxisLayer(l_4ch_2b, pool_size=(2,), axis=(4,))
    l_4ch_2r = nn.layers.DimshuffleLayer(l_4ch_2m, (0,2,3,1,4))
    l_4ch_2r = reshape(l_4ch_2r, (-1, ) + l_4ch_2m.output_shape[1:2] + l_4ch_2m.output_shape[-1:])
    l_4ch_d1 = nn.layers.DenseLayer(l_4ch_2r, num_units=64, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_4ch_d2 = nn.layers.DenseLayer(l_4ch_d1, num_units=64, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    l_4ch_mu = nn.layers.DenseLayer(l_4ch_d2, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_4ch_sigma = nn.layers.DenseLayer(l_4ch_d2, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    l_4ch_mu = reshape(l_4ch_mu, (-1, ) + l_4ch.output_shape[-3:-1])
    l_4ch_sigma = reshape(l_4ch_sigma, (-1, ) + l_4ch.output_shape[-3:-1])


    l_2ch_2a = deep_learning_layers.ConvolutionOver2DAxisLayer(l_2chr, num_filters=16, filter_size=(3, 3),
                                                             axis=(3,4), channel=1,
                                                             pad=(1,1),
                                                             W=nn.init.Orthogonal(),
                                                             b=nn.init.Constant(0.0),
                                                             )
    l_2ch_2b = deep_learning_layers.ConvolutionOver2DAxisLayer(l_2ch_2a, num_filters=16, filter_size=(3, 3),
                                                             axis=(3,4), channel=1,
                                                             pad=(1,1),
                                                             W=nn.init.Orthogonal(),
                                                             b=nn.init.Constant(0.0),
                                                             )
    l_2ch_2m = deep_learning_layers.MaxPoolOverAxisLayer(l_2ch_2b, pool_size=(2,), axis=(4,))

    l_2ch_2r = nn.layers.DimshuffleLayer(l_2ch_2m, (0,2,3,1,4))
    l_2ch_2r = reshape(l_2ch_2r, (-1, ) + l_2ch_2m.output_shape[1:2] + l_2ch_2m.output_shape[-1:])
    l_2ch_d1 = nn.layers.DenseLayer(l_2ch_2r, num_units=64, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_2ch_d2 = nn.layers.DenseLayer(l_2ch_d1, num_units=64, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    l_2ch_mu = nn.layers.DenseLayer(l_2ch_d2, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_2ch_sigma = nn.layers.DenseLayer(l_2ch_d2, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    l_2ch_mu = reshape(l_2ch_mu, (-1, ) + l_2ch.output_shape[-3:-1])
    l_2ch_sigma = reshape(l_2ch_sigma, (-1, ) + l_2ch.output_shape[-3:-1])

    l_sys_and_dia = layers.IraLayer(l_4ch_mu, l_4ch_sigma, l_2ch_mu, l_2ch_sigma)

    l_systole = nn.layers.SliceLayer(l_sys_and_dia, indices=0, axis=2)
    l_diastole = nn.layers.SliceLayer(l_sys_and_dia, indices=1, axis=2)

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

