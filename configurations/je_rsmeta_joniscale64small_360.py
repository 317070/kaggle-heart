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
num_epochs_train = 100 

# - learning rate and method
base_lr = 0.01
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

# Input sizes
image_size = 64
nr_slices = 8
data_sizes = {
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


# Architecture
def build_model():

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:randomslices"]

    l0 = nn.layers.InputLayer(input_size)

    # PREPROCESS SLICES SEPERATELY
    # Convolutional layers and some dense layers are defined in a submodel
    l0_slices = nn.layers.ReshapeLayer(l0, (-1, [2], [3], [4]))

    import je_ss_jonisc64small_360
    submodel = je_ss_jonisc64small_360.build_model(l0_slices)

    # Systole Dense layers
    ldsys2 = submodel["meta_outputs"]["systole"]
    # Diastole Dense layers
    lddia2 = submodel["meta_outputs"]["diastole"]

    # AGGREGATE SLICES PER PATIENT
    # Systole
    ldsys_pat_in = nn.layers.ReshapeLayer(ldsys2, (-1, nr_slices, [1]))

    input_gate_sys = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    forget_gate_sys = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), b=nn.init.Constant(5.0))
    output_gate_sys = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    cell_sys = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), W_cell=None, nonlinearity=nn.nonlinearities.tanh)

    ldsys_lstm = nn.layers.LSTMLayer(ldsys_pat_in, num_units=256,
                                     ingate=input_gate_sys, forgetgate=forget_gate_sys,
                                     cell=cell_sys, outgate=output_gate_sys,
                                     peepholes=False, precompute_input=False,
                                     grad_clipping=5, only_return_final=True,
                                     learn_init=True,)
 
    ldsys_lstm_drop = nn.layers.dropout(ldsys_lstm, p=0.5)

    ldsys3mu = nn.layers.DenseLayer(ldsys_lstm_drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(200.0), nonlinearity=None)
    ldsys3sigma = nn.layers.DenseLayer(ldsys_lstm_drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(100.0), nonlinearity=lb_softplus(3))
    ldsys3musigma = nn.layers.ConcatLayer([ldsys3mu, ldsys3sigma], axis=1)

    l_systole = layers.MuSigmaErfLayer(ldsys3musigma)

    # Diastole
    lddia_pat_in = nn.layers.ReshapeLayer(lddia2, (-1, nr_slices, [1]))

    input_gate_dia = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    forget_gate_dia = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), b=nn.init.Constant(5.0))
    output_gate_dia = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
    cell_dia = nn.layers.Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), W_cell=None, nonlinearity=nn.nonlinearities.tanh)

    lddia_lstm = nn.layers.LSTMLayer(lddia_pat_in, num_units=256,
                                     ingate=input_gate_dia, forgetgate=forget_gate_dia,
                                     cell=cell_dia, outgate=output_gate_dia,
                                     peepholes=False, precompute_input=False,
                                     grad_clipping=5, only_return_final=True,
                                     learn_init=True,)
 
    lddia_lstm_drop = nn.layers.dropout(lddia_lstm, p=0.5)

    lddia3mu = nn.layers.DenseLayer(lddia_lstm_drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(200.0), nonlinearity=None)
    lddia3sigma = nn.layers.DenseLayer(lddia_lstm_drop, num_units=1, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(100.0), nonlinearity=lb_softplus(3))
    lddia3musigma = nn.layers.ConcatLayer([lddia3mu, lddia3sigma], axis=1)

    l_diastole = layers.MuSigmaErfLayer(lddia3musigma)

    submodels = [submodel]
    return {
        "inputs":{
            "sliced:data:randomslices": l0
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": dict(
            {
            ldsys3mu: l2_weight_out,
            ldsys3sigma: l2_weight_out,
            lddia3mu: l2_weight_out,
            lddia3sigma: l2_weight_out,},
            **{
                k: v
                for d in [model["regularizable"] for model in submodels if "regularizable" in model]
                for k, v in d.items() }
        ),
        "pretrained":{
            je_ss_jonisc64small_360.__name__: submodel["outputs"],
        }
    }

