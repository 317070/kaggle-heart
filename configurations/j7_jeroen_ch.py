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
batch_size = 4
sunny_batch_size = 4
batches_per_chunk = 32
num_epochs_train = 150 

# - learning rate and method
base_lr = 0.0001
learning_rate_schedule = {
    0: base_lr,
    8*num_epochs_train/10: base_lr/10,
    19*num_epochs_train/20: base_lr/100,
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
    import glob
    return folders

# Input sizes
image_size = 64
nr_slices = 22
data_sizes = {
    "sliced:data:sax": (batch_size, nr_slices, 30, image_size, image_size),
    "sliced:data:sax:locations": (batch_size, nr_slices),
    "sliced:data:sax:is_not_padded": (batch_size, nr_slices),
    "sliced:data:randomslices": (batch_size, nr_slices, 30, image_size, image_size),
    "sliced:data:singleslice:2ch": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:4ch": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
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

    import j6_2ch_gauss, j6_4ch_gauss
    meta_2ch = j6_2ch_gauss.build_model()
    meta_4ch = j6_4ch_gauss.build_model()

    l_meta_2ch_systole = nn.layers.DenseLayer(meta_2ch["meta_outputs"]["systole"], num_units=64, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_meta_2ch_diastole = nn.layers.DenseLayer(meta_2ch["meta_outputs"]["diastole"], num_units=64, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    l_meta_4ch_systole = nn.layers.DenseLayer(meta_4ch["meta_outputs"]["systole"], num_units=64, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_meta_4ch_diastole = nn.layers.DenseLayer(meta_4ch["meta_outputs"]["diastole"], num_units=64, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

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

    import je_ss_jonisc64small_360_gauss_longer
    submodel = je_ss_jonisc64small_360_gauss_longer.build_model(l0_slices)

    # Systole Dense layers
    l_sys_mu = submodel["meta_outputs"]["systole:mu"]
    l_sys_sigma = submodel["meta_outputs"]["systole:sigma"]
    l_sys_meta =  submodel["meta_outputs"]["systole"]
    # Diastole Dense layers
    l_dia_mu = submodel["meta_outputs"]["diastole:mu"]
    l_dia_sigma = submodel["meta_outputs"]["diastole:sigma"]
    l_dia_meta =  submodel["meta_outputs"]["diastole"]

    # AGGREGATE SLICES PER PATIENT
    l_scaled_slice_locations = layers.TrainableScaleLayer(lin_slice_locations, scale=nn.init.Constant(0.1), trainable=False)

    # Systole
    l_pat_sys_ss_mu = nn.layers.ReshapeLayer(l_sys_mu, (-1, nr_slices))
    l_pat_sys_ss_sigma = nn.layers.ReshapeLayer(l_sys_sigma, (-1, nr_slices))
    l_pat_sys_aggr_mu_sigma = layers.JeroenLayer([l_pat_sys_ss_mu, l_pat_sys_ss_sigma, lin_slice_mask, l_scaled_slice_locations], rescale_input=100.)

    l_systole = layers.MuSigmaErfLayer(l_pat_sys_aggr_mu_sigma)

    l_sys_meta = nn.layers.DenseLayer(nn.layers.ReshapeLayer(l_sys_meta, (-1, nr_slices, 512)), num_units=64, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    l_meta_systole = nn.layers.ConcatLayer([l_meta_2ch_systole, l_meta_4ch_systole, l_sys_meta])
    l_weights = nn.layers.DenseLayer(l_meta_systole, num_units=512, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_weights = nn.layers.DenseLayer(l_weights, num_units=3, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    systole_output = layers.WeightedMeanLayer(l_weights, [l_systole, meta_2ch["outputs"]["systole"], meta_4ch["outputs"]["systole"]])

    # Diastole
    l_pat_dia_ss_mu = nn.layers.ReshapeLayer(l_dia_mu, (-1, nr_slices))
    l_pat_dia_ss_sigma = nn.layers.ReshapeLayer(l_dia_sigma, (-1, nr_slices))
    l_pat_dia_aggr_mu_sigma = layers.JeroenLayer([l_pat_dia_ss_mu, l_pat_dia_ss_sigma, lin_slice_mask, l_scaled_slice_locations], rescale_input=100.)

    l_diastole = layers.MuSigmaErfLayer(l_pat_dia_aggr_mu_sigma)

    l_dia_meta = nn.layers.DenseLayer(nn.layers.ReshapeLayer(l_dia_meta, (-1, nr_slices, 512)), num_units=64, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    l_meta_diastole = nn.layers.ConcatLayer([l_meta_2ch_diastole, l_meta_4ch_diastole, l_dia_meta])
    l_weights = nn.layers.DenseLayer(l_meta_diastole, num_units=512, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)
    l_weights = nn.layers.DenseLayer(l_weights, num_units=3, W=nn.init.Orthogonal(), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.identity)
    diastole_output = layers.WeightedMeanLayer(l_weights, [l_diastole, meta_2ch["outputs"]["diastole"], meta_4ch["outputs"]["diastole"]])

    submodels = [submodel, meta_2ch, meta_4ch]


    return {
        "inputs":dict({
            "sliced:data:sax": l0,
            "sliced:data:sax:is_not_padded": lin_slice_mask,
            "sliced:data:sax:locations": lin_slice_locations,
        }, **{ k: v for d in [model["inputs"] for model in [meta_2ch, meta_4ch]]
               for k, v in d.items() }
        ),
        "outputs": {
            "systole": systole_output,
            "diastole": diastole_output,
        },
        "regularizable": dict(
            {},
            **{
                k: v
                for d in [model["regularizable"] for model in submodels if "regularizable" in model]
                for k, v in d.items() }
        ),
        "pretrained":{
            je_ss_jonisc64small_360_gauss_longer.__name__: submodel["outputs"],
            j6_2ch_gauss.__name__: meta_2ch["outputs"],
            j6_4ch_gauss.__name__: meta_4ch["outputs"],
        },
        #"cutoff_gradients": [
        #] + [ v for d in [model["meta_outputs"] for model in [meta_2ch, meta_4ch] if "meta_outputs" in model]
        #       for v in d.values() ]
    }






