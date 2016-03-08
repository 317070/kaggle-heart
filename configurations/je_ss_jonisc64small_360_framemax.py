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
batch_size = 4
sunny_batch_size = 4
batches_per_chunk = 16 * 8
AV_SLICE_PER_PAT = 11
num_epochs_train = 100 * AV_SLICE_PER_PAT

# - learning rate and method
base_lr = .001
learning_rate_schedule = {
    0: base_lr,
    num_epochs_train*9/10: base_lr/10,
}
momentum = 0.9
build_updates = updates.build_adam_updates

# Preprocessing stuff
cleaning_processes = [
    preprocess.set_upside_up,]
cleaning_processes_post = [
    functools.partial(preprocess.normalize_contrast_zmuv, z=2)]

augmentation_params = {
    "rotation": (-180, 180),
    "shear": (0, 0),
    "translation": (-8, 8),
    "flip_vert": (0, 1),
    "roll_time": (0, 0),
    "flip_time": (0, 0),
}

use_hough_roi = True  # use roi to center patches
preprocess_train = functools.partial(  # normscale_resize_and_augment has a bug
    preprocess.preprocess_normscale,
    normscale_resize_and_augment_function=functools.partial(
        image_transform.normscale_resize_and_augment_2,
        normalised_patch_size=(64,64)))
preprocess_validation = functools.partial(preprocess_train, augment=False)
preprocess_test = preprocess_train

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
NR_FRAMES = 30
data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, NR_FRAMES, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:ax": (batch_size, NR_FRAMES, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
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

def lb_softplus(lb):
    return lambda x: nn.nonlinearities.softplus(x) + lb


def build_model(input_layer = None):

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:singleslice"]

    if input_layer:
        l0 = input_layer
    else:
        l0 = nn.layers.InputLayer(input_size)

    # Reshape to framemodel
    l0_slices = nn.layers.ReshapeLayer(l0, (-1, 1, [2], [3]))

    l1a = nn.layers.dnn.Conv2DDNNLayer(l0_slices,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=16, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1b = nn.layers.dnn.Conv2DDNNLayer(l1a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=16, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1 = nn.layers.dnn.MaxPool2DDNNLayer(l1b, pool_size=(2,2), stride=(2,2))

    l2a = nn.layers.dnn.Conv2DDNNLayer(l1,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2b = nn.layers.dnn.Conv2DDNNLayer(l2a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2 = nn.layers.dnn.MaxPool2DDNNLayer(l2b, pool_size=(2,2), stride=(2,2))

    l3a = nn.layers.dnn.Conv2DDNNLayer(l2,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3b = nn.layers.dnn.Conv2DDNNLayer(l3a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3c = nn.layers.dnn.Conv2DDNNLayer(l3b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3 = nn.layers.dnn.MaxPool2DDNNLayer(l3c, pool_size=(2,2), stride=(2,2))

    l4a = nn.layers.dnn.Conv2DDNNLayer(l3,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4b = nn.layers.dnn.Conv2DDNNLayer(l4a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4c = nn.layers.dnn.Conv2DDNNLayer(l4b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4 = nn.layers.dnn.MaxPool2DDNNLayer(l4c, pool_size=(2,2), stride=(2,2))

    l5a = nn.layers.dnn.Conv2DDNNLayer(l4,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5b = nn.layers.dnn.Conv2DDNNLayer(l5a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5c = nn.layers.dnn.Conv2DDNNLayer(l5b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5 = nn.layers.dnn.MaxPool2DDNNLayer(l5c, pool_size=(2,2), stride=(2,2))

    l5drop = nn.layers.dropout(l5, p=0.0)
    ld1 = nn.layers.DenseLayer(l5drop, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ld1drop = nn.layers.dropout(ld1, p=0.0)
    ld2 = nn.layers.DenseLayer(ld1drop, num_units=512, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ld2drop = nn.layers.dropout(ld2, p=0.0)

    ld3mu = nn.layers.DenseLayer(ld2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(200.0), nonlinearity=None)
    ld3sigma = nn.layers.DenseLayer(ld2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(50.0), nonlinearity=lb_softplus(3))
    ld3musigma = nn.layers.ConcatLayer([ld3mu, ld3sigma], axis=1)

    # Reshape back to slicemodel
    ld3musigma_slices = nn.layers.ReshapeLayer(ld3musigma, (-1, NR_FRAMES, 2))

    l_systole_musigma = layers.ArgmaxAndMaxLayer(ld3musigma_slices, 'min')
    l_systole = layers.MuSigmaErfLayer(l_systole_musigma)

    l_diastole_musigma = layers.ArgmaxAndMaxLayer(ld3musigma_slices, 'max')
    l_diastole = layers.MuSigmaErfLayer(l_diastole_musigma)


    return {
        "inputs":{
            "sliced:data:singleslice": l0
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": {
            ld1: l2_weight,
            ld2: l2_weight,
            ld3mu: l2_weight_out,
            ld3musigma: l2_weight_out,
        }
    }

