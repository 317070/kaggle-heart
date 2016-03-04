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
batch_size = 2
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


init = nn.init.Orthogonal()

rnn_layer = functools.partial(nn.layers.RecurrentLayer,
    W_in_to_hid=init,
    W_hid_to_hid=init,
    b=nn.init.Constant(0.1),
    nonlinearity=nn.nonlinearities.rectify,
    hid_init=nn.init.Constant(0.),
    backwards=False,
    learn_init=True,
    gradient_steps=-1,
    grad_clipping=False,
    unroll_scan=False,
    precompute_input=False,
    mask_input=None,
    only_return_final=True)


# Architecture
def build_model(input_layer = None):

    #################
    # Regular model #
    #################

    l_4ch = nn.layers.InputLayer(data_sizes["sliced:data:chanzoom:4ch"])
    l_2ch = nn.layers.InputLayer(data_sizes["sliced:data:chanzoom:2ch"])

    # Add an axis to concatenate over later
    l_4chr = nn.layers.ReshapeLayer(l_4ch, (batch_size, 1, ) + l_4ch.output_shape[1:])
    l_2chr = nn.layers.ReshapeLayer(l_2ch, (batch_size, 1, ) + l_2ch.output_shape[1:])
    
    # Cut the images in half, flip the left ones
    l_4ch_left = nn.layers.SliceLayer(l_4chr, indices=slice(image_size//2-1, None, -1), axis=-1)
    l_4ch_right = nn.layers.SliceLayer(l_4chr, indices=slice(image_size//2, None, 1), axis=-1)
    l_2ch_left = nn.layers.SliceLayer(l_2chr, indices=slice(image_size//2-1, None, -1), axis=-1)
    l_2ch_right = nn.layers.SliceLayer(l_2chr, indices=slice(image_size//2, None, 1), axis=-1)

    # Concatenate over second axis
    l_24lr = nn.layers.ConcatLayer([l_4ch_left, l_4ch_right, l_2ch_left, l_2ch_right], axis=1)
    # b, 4, t, h, w

    # Subsample frames
    SUBSAMPLING_FACTOR = 2
    nr_subsampled_frames = nr_frames // SUBSAMPLING_FACTOR
    l_24lr_ss = nn.layers.SliceLayer(l_24lr, indices=slice(None, None, SUBSAMPLING_FACTOR), axis=2)

    # Move frames and halves to batch, process them all in the same way, add channel axis
    l_halves = nn.layers.ReshapeLayer(l_24lr_ss, (batch_size * 4 * nr_subsampled_frames, 1, image_size, image_size//2))

    # First, do some convolutions in all directions
    num_channels = 64
    l1a = nn.layers.dnn.Conv2DDNNLayer(l_halves,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=num_channels, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1b = nn.layers.dnn.Conv2DDNNLayer(l1a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=num_channels, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    
    # Then, put an rnn over the last axis
    l1_shuffle = nn.layers.DimshuffleLayer(l1b, pattern=(0, 2, 3, 1))
    l1_r = nn.layers.ReshapeLayer(l1_shuffle, (batch_size * 4 * nr_subsampled_frames * image_size, image_size//2, num_channels))

    l_rnn = rnn_layer(l1_r, 1024)

    ld3mu = nn.layers.DenseLayer(l_rnn, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(16.0), nonlinearity=None)
    ld3sigma = nn.layers.DenseLayer(l_rnn, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(4.0), nonlinearity=lb_softplus(.01))
    ld3musigma = nn.layers.ConcatLayer([ld3mu, ld3sigma], axis=1)

    # Get the four halves back
    l_24lr_musigma = nn.layers.ReshapeLayer(ld3musigma, (batch_size, 4, nr_subsampled_frames, image_size, 2))
    l_24lr_musigma_shuffle = nn.layers.DimshuffleLayer(l_24lr_musigma, pattern=(0, 2, 1, 3, 4))
    l_24lr_musigma_re = nn.layers.ReshapeLayer(l_24lr_musigma_shuffle, (batch_size * nr_subsampled_frames, 4, image_size, 2))

    l_4ch_left_musigma = nn.layers.SliceLayer(l_24lr_musigma_re, indices=0, axis=1)   
    l_4ch_right_musigma = nn.layers.SliceLayer(l_24lr_musigma_re, indices=1, axis=1)   
    l_2ch_left_musigma = nn.layers.SliceLayer(l_24lr_musigma_re, indices=2, axis=1)   
    l_2ch_right_musigma = nn.layers.SliceLayer(l_24lr_musigma_re, indices=3, axis=1)

    l_4ch_musigma = layers.SumGaussLayer([l_4ch_left_musigma, l_4ch_right_musigma])
    l_2ch_musigma = layers.SumGaussLayer([l_2ch_left_musigma, l_2ch_right_musigma])

    l_musigma_frames = layers.IraLayerNoTime(l_4ch_musigma, l_2ch_musigma)

    # Minmax over time
    print l_musigma_frames.output_shape
    l_musigmas = nn.layers.ReshapeLayer(l_musigma_frames, (-1, nr_subsampled_frames, 2))
    l_musigma_sys = layers.ArgmaxAndMaxLayer(l_musigmas, mode='min')
    l_musigma_dia = layers.ArgmaxAndMaxLayer(l_musigmas, mode='max')

    l_systole = layers.MuSigmaErfLayer(l_musigma_sys)
    l_diastole = layers.MuSigmaErfLayer(l_musigma_dia)
 
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

