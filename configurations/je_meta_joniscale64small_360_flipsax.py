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

dump_network_loaded_data = False

# Training (schedule) parameters
# - batch sizes
batch_size = 16
sunny_batch_size = 4
batches_per_chunk = 16
num_epochs_train = 100 

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
    "flip_sax": (0, 1),
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

    l0 = nn.layers.InputLayer(input_size)
    lin_slice_mask = nn.layers.InputLayer(input_size_mask)
    lin_slice_locations = nn.layers.InputLayer(input_size_locations)

    # PREPROCESS SLICES SEPERATELY
    # Convolutional layers and some dense layers are defined in a submodel
    l0_slices = nn.layers.ReshapeLayer(l0, (-1, [2], [3], [4]))

    import je_ss_jonisc64small_360
    submodel = je_ss_jonisc64small_360.build_model(l0_slices)

    # Systole Dense layers
    ldsysout = submodel["outputs"]["systole"]
    # Diastole Dense layers
    lddiaout = submodel["outputs"]["diastole"]

    # AGGREGATE SLICES PER PATIENT
    # Systole
    ldsys_pat_in = nn.layers.ReshapeLayer(ldsysout, (-1, nr_slices, [1]))

    ldsys_rnn = rnn_layer(ldsys_pat_in, num_units=256, mask_input=lin_slice_mask)
 
#    ldsys_rnn_drop = nn.layers.dropout(ldsys_rnn, p=0.5)

    ldsys3 = nn.layers.DenseLayer(ldsys_rnn, num_units=600, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)
    ldsys3drop = nn.layers.dropout(ldsys3, p=0.5)  # dropout at the output might encourage adjacent neurons to correllate
    ldsys3dropnorm = layers.NormalisationLayer(ldsys3drop)
    l_systole = layers.CumSumLayer(ldsys3dropnorm)

    # Diastole
    lddia_pat_in = nn.layers.ReshapeLayer(lddiaout, (-1, nr_slices, [1]))

    lddia_rnn = rnn_layer(lddia_pat_in, num_units=256, mask_input=lin_slice_mask)
 
#    lddia_rnn_drop = nn.layers.dropout(lddia_rnn, p=0.5)
    lddia3 = nn.layers.DenseLayer(lddia_rnn, num_units=600, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)
    lddia3drop = nn.layers.dropout(lddia3, p=0.5)  # dropout at the output might encourage adjacent neurons to correllate
    lddia3dropnorm = layers.NormalisationLayer(lddia3drop)
    l_diastole = layers.CumSumLayer(lddia3dropnorm)


    submodels = [submodel]
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
        "regularizable": dict(
            {
            lddia3: l2_weight_out,
            ldsys3: l2_weight_out,},
            **{
                k: v
                for d in [model["regularizable"] for model in submodels if "regularizable" in model]
                for k, v in d.items() }
        ),
        "pretrained":{
            je_ss_jonisc64small_360.__name__: submodel["outputs"],
        }
    }

