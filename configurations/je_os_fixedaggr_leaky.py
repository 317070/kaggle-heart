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
save_every = 20
restart_from_save = False

dump_network_loaded_data = False

# Training (schedule) parameters
# - batch sizes
batch_size = 8
sunny_batch_size = 4
batches_per_chunk = 16
num_epochs_train = 200

# - learning rate and method
base_lr = 0.0001
learning_rate_schedule = {
    0: base_lr,
    9*num_epochs_train/10: base_lr/10,
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

use_hough_roi = True
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

def filter_samples(folders):
    # don't use patients who don't have mre than 6 slices
    return [
        folder for folder in folders
        if data_loader.compute_nr_slices(folder) > 6]

# Input sizes
image_size = 64
nr_slices = 20
data_sizes = {
    "sliced:data:sax": (batch_size, nr_slices, 30, image_size, image_size),
    "sliced:data:sax:locations": (batch_size, nr_slices),
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
    nonlinearity=nn.nonlinearities.very_leaky_rectify,
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

    l0 = nn.layers.InputLayer(input_size)
    lin_slice_mask = nn.layers.InputLayer(input_size_mask)
    lin_slice_locations = nn.layers.InputLayer(input_size_locations)

    # PREPROCESS SLICES SEPERATELY
    # Convolutional layers and some dense layers are defined in a submodel
    l0_slices = nn.layers.ReshapeLayer(l0, (-1, [2], [3], [4]))

    l1a = nn.layers.dnn.Conv2DDNNLayer(l0_slices,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l1b = nn.layers.dnn.Conv2DDNNLayer(l1a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l1 = nn.layers.dnn.MaxPool2DDNNLayer(l1b, pool_size=(2,2), stride=(2,2))

    l2a = nn.layers.dnn.Conv2DDNNLayer(l1,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l2b = nn.layers.dnn.Conv2DDNNLayer(l2a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l2 = nn.layers.dnn.MaxPool2DDNNLayer(l2b, pool_size=(2,2), stride=(2,2))

    l3a = nn.layers.dnn.Conv2DDNNLayer(l2,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l3b = nn.layers.dnn.Conv2DDNNLayer(l3a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l3c = nn.layers.dnn.Conv2DDNNLayer(l3b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l3 = nn.layers.dnn.MaxPool2DDNNLayer(l3c, pool_size=(2,2), stride=(2,2))

    l4a = nn.layers.dnn.Conv2DDNNLayer(l3,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l4b = nn.layers.dnn.Conv2DDNNLayer(l4a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l4c = nn.layers.dnn.Conv2DDNNLayer(l4b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l4 = nn.layers.dnn.MaxPool2DDNNLayer(l4c, pool_size=(2,2), stride=(2,2))

    l5a = nn.layers.dnn.Conv2DDNNLayer(l4,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l5b = nn.layers.dnn.Conv2DDNNLayer(l5a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l5c = nn.layers.dnn.Conv2DDNNLayer(l5b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l5 = nn.layers.dnn.MaxPool2DDNNLayer(l5c, pool_size=(2,2), stride=(2,2))

    # Systole Dense layers
    ldsys1 = nn.layers.DenseLayer(l5, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.very_leaky_rectify)

    ldsys1drop = nn.layers.dropout(ldsys1, p=0.5)
    ldsys2 = nn.layers.DenseLayer(ldsys1drop, num_units=512, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.very_leaky_rectify)

    ldsys2drop = nn.layers.dropout(ldsys2, p=0.5)
    l_sys_mu = nn.layers.DenseLayer(ldsys2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(10.0), nonlinearity=None)
    l_sys_sigma = nn.layers.DenseLayer(ldsys2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(3.), nonlinearity=lb_softplus(.1))

    # Diastole Dense layers
    lddia1 = nn.layers.DenseLayer(l5, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.very_leaky_rectify)

    lddia1drop = nn.layers.dropout(lddia1, p=0.5)
    lddia2 = nn.layers.DenseLayer(lddia1drop, num_units=512, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.very_leaky_rectify)

    lddia2drop = nn.layers.dropout(lddia2, p=0.5)
    l_dia_mu = nn.layers.DenseLayer(lddia2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(10.0), nonlinearity=None)
    l_dia_sigma = nn.layers.DenseLayer(lddia2drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(3.), nonlinearity=lb_softplus(.1))

    # AGGREGATE SLICES PER PATIENT
    l_scaled_slice_locations = layers.TrainableScaleLayer(lin_slice_locations, scale=nn.init.Constant(0.1), trainable=False)

    # Systole
    l_pat_sys_ss_mu = nn.layers.ReshapeLayer(l_sys_mu, (-1, nr_slices))
    l_pat_sys_ss_sigma = nn.layers.ReshapeLayer(l_sys_sigma, (-1, nr_slices))
    l_pat_sys_aggr_mu_sigma = layers.JeroenLayer([l_pat_sys_ss_mu, l_pat_sys_ss_sigma, lin_slice_mask, l_scaled_slice_locations], rescale_input=1.)

    l_systole = layers.MuSigmaErfLayer(l_pat_sys_aggr_mu_sigma)

    # Diastole
    l_pat_dia_ss_mu = nn.layers.ReshapeLayer(l_dia_mu, (-1, nr_slices))
    l_pat_dia_ss_sigma = nn.layers.ReshapeLayer(l_dia_sigma, (-1, nr_slices))
    l_pat_dia_aggr_mu_sigma = layers.JeroenLayer([l_pat_dia_ss_mu, l_pat_dia_ss_sigma, lin_slice_mask, l_scaled_slice_locations], rescale_input=1.)

    l_diastole = layers.MuSigmaErfLayer(l_pat_dia_aggr_mu_sigma)

    return {
        "inputs":{
            "sliced:data:sax": l0,
            "sliced:data:sax:is_not_padded": lin_slice_mask,
            "sliced:data:sax:locations": lin_slice_locations,
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": {
            ldsys1: l2_weight,
            ldsys2: l2_weight,
            l_sys_mu: l2_weight_out,
            l_sys_sigma: l2_weight_out,
            lddia1: l2_weight,
            lddia2: l2_weight,
            l_dia_mu: l2_weight_out,
            l_dia_sigma: l2_weight_out,
        },
    }

