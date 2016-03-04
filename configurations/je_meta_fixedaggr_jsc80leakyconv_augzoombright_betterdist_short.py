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
validate_every = 20
validate_train_set = True
save_every = 5
restart_from_save = False

dump_network_loaded_data = False

# Training (schedule) parameters
# - batch sizes
batch_size = 1
sunny_batch_size = 4
batches_per_chunk = 32 *4
num_epochs_train = 62

# - learning rate and method
base_lr = 0.00003
learning_rate_schedule = {
    0: base_lr,
    45: base_lr/10,
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
    "zoom_x": (.75, 1.25),
    "zoom_y": (.75, 1.25),
    "change_brightness": (-0.3, 0.3),
}

augmentation_params_test = {
    "rotation": (-180, 180),
    "shear": (0, 0),
    "translation": (-8, 8),
    "flip_vert": (0, 1),
    "roll_time": (0, 0),
    "flip_time": (0, 0),
    "zoom_x": (.80, 1.20),
    "zoom_y": (.80, 1.20),
    "change_brightness": (-0.2, 0.2),
}

use_hough_roi = True
preprocess_train = functools.partial(  # normscale_resize_and_augment has a bug
    preprocess.preprocess_normscale,
    normscale_resize_and_augment_function=functools.partial(
        image_transform.normscale_resize_and_augment_2,
        normalised_patch_size=(80,80)))
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

def filter_samples(folders):
    # don't use patients who don't have mre than 6 slices
    return [
        folder for folder in folders
        if data_loader.compute_nr_slices(folder) > 6]

# Input sizes
image_size = 64
nr_slices = 22
data_sizes = {
    "sliced:data:sax": (batch_size, nr_slices, 30, image_size, image_size),
    "sliced:data:sax:locations": (batch_size, nr_slices),
    "sliced:data:sax:distances": (batch_size, nr_slices),
    "sliced:data:sax:is_not_padded": (batch_size, nr_slices),
    "sliced:data:randomslices": (batch_size, nr_slices, 30, image_size, image_size),
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size),
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size),
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size),
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size),
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
test_time_augmentations = 100  # More augmentations since a we only use single slices
tta_average_method = lambda x: np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))


# nonlinearity putting a lower bound on it's output
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
    precompute_input=False)



# Architecture
def build_model():

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:sax"]
    input_size_mask = data_sizes["sliced:data:sax:is_not_padded"]
    input_size_locations = data_sizes["sliced:data:sax:locations"]
    input_size_distances = data_sizes["sliced:data:sax:distances"]

    l0 = nn.layers.InputLayer(input_size)
    lin_slice_mask = nn.layers.InputLayer(input_size_mask)
    lin_slice_locations = nn.layers.InputLayer(input_size_locations)
    lin_slice_distances = nn.layers.InputLayer(input_size_distances)

    # PREPROCESS SLICES SEPERATELY
    # Convolutional layers and some dense layers are defined in a submodel
    l0_slices = nn.layers.ReshapeLayer(l0, (-1, [2], [3], [4]))

    import je_ss_jonisc80_leaky_convroll_augzoombright
    submodel = je_ss_jonisc80_leaky_convroll_augzoombright.build_model(l0_slices)

    # Systole Dense layers
    l_sys_mu = submodel["meta_outputs"]["systole:mu"]
    l_sys_sigma = submodel["meta_outputs"]["systole:sigma"]
    # Diastole Dense layers
    l_dia_mu = submodel["meta_outputs"]["diastole:mu"]
    l_dia_sigma = submodel["meta_outputs"]["diastole:sigma"]

    # AGGREGATE SLICES PER PATIENT
    l_scaled_slice_locations = layers.TrainableScaleLayer(lin_slice_locations, scale=nn.init.Constant(0.1), trainable=False)
    l_scaled_slice_distances = layers.TrainableScaleLayer(lin_slice_distances, scale=nn.init.Constant(0.1), trainable=False)

    # Systole
    l_pat_sys_ss_mu = nn.layers.ReshapeLayer(l_sys_mu, (-1, nr_slices))
    l_pat_sys_ss_sigma = nn.layers.ReshapeLayer(l_sys_sigma, (-1, nr_slices))
    l_pat_sys_aggr_mu_sigma = layers.JeroenLayerDists([l_pat_sys_ss_mu, l_pat_sys_ss_sigma, lin_slice_mask, l_scaled_slice_distances], rescale_input=100.)

    l_systole = layers.MuSigmaErfLayer(l_pat_sys_aggr_mu_sigma)

    # Diastole
    l_pat_dia_ss_mu = nn.layers.ReshapeLayer(l_dia_mu, (-1, nr_slices))
    l_pat_dia_ss_sigma = nn.layers.ReshapeLayer(l_dia_sigma, (-1, nr_slices))
    l_pat_dia_aggr_mu_sigma = layers.JeroenLayerDists([l_pat_dia_ss_mu, l_pat_dia_ss_sigma, lin_slice_mask, l_scaled_slice_distances], rescale_input=100.)

    l_diastole = layers.MuSigmaErfLayer(l_pat_dia_aggr_mu_sigma)


    submodels = [submodel]
    return {
        "inputs":{
            "sliced:data:sax": l0,
            "sliced:data:sax:is_not_padded": lin_slice_mask,
            "sliced:data:sax:locations": lin_slice_locations,
            "sliced:data:sax:distances": lin_slice_distances,
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": dict(
            {},
            **{
                k: v
                for d in [model["regularizable"] for model in submodels if "regularizable" in model]
                for k, v in d.items() }
        ),
        "pretrained":{
            je_ss_jonisc80_leaky_convroll_augzoombright.__name__: submodel["outputs"],
        }
    }






