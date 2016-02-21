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
batch_size = 32
sunny_batch_size = 4
batches_per_chunk = 16
AV_SLICE_PER_PAT = 1
num_epochs_train = 175 * AV_SLICE_PER_PAT

# - learning rate and method
base_lr = .0001
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
data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:2ch": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:4ch": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sliced:meta:PatientAge": (batch_size, 1),
    "sliced:meta:PatientSex": (batch_size, 1),
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
test_time_augmentations = 200  # More augmentations since a we only use single slices
tta_average_method = lambda x: np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))

# Architecture
def build_model():

    #import here, such that our global variables are not overridden!
    import j6_2ch_128mm, j6_4ch

    meta_2ch = j6_2ch_128mm.build_model()
    meta_4ch = j6_4ch.build_model()

    l_age = nn.layers.InputLayer(data_sizes["sliced:meta:PatientAge"])
    l_sex = nn.layers.InputLayer(data_sizes["sliced:meta:PatientSex"])

    l_meta_2ch_systole = meta_2ch["meta_outputs"]["systole"]
    l_meta_2ch_diastole = meta_2ch["meta_outputs"]["diastole"]

    l_meta_4ch_systole = meta_4ch["meta_outputs"]["systole"]
    l_meta_4ch_diastole = meta_4ch["meta_outputs"]["diastole"]

    l_meta_systole = nn.layers.ConcatLayer([l_age, l_sex, l_meta_2ch_systole, l_meta_4ch_systole])
    l_meta_diastole = nn.layers.ConcatLayer([l_age, l_sex, l_meta_2ch_diastole, l_meta_4ch_diastole])

    ldsys1 = nn.layers.DenseLayer(l_meta_systole, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys1drop = nn.layers.dropout(ldsys1, p=0.5)
    ldsys2 = nn.layers.DenseLayer(ldsys1drop, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys2drop = nn.layers.dropout(ldsys2, p=0.5)
    ldsys3 = nn.layers.DenseLayer(ldsys2drop, num_units=600, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)

    ldsys3drop = nn.layers.dropout(ldsys3, p=0.5)  # dropout at the output might encourage adjacent neurons to correllate
    ldsys3dropnorm = layers.NormalisationLayer(ldsys3drop)
    l_systole = layers.CumSumLayer(ldsys3dropnorm)


    lddia1 = nn.layers.DenseLayer(l_meta_diastole, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia1drop = nn.layers.dropout(lddia1, p=0.5)
    lddia2 = nn.layers.DenseLayer(lddia1drop, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia2drop = nn.layers.dropout(lddia2, p=0.5)
    lddia3 = nn.layers.DenseLayer(lddia2drop, num_units=600, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)

    lddia3drop = nn.layers.dropout(lddia3, p=0.5)  # dropout at the output might encourage adjacent neurons to correllate
    lddia3dropnorm = layers.NormalisationLayer(lddia3drop)
    l_diastole = layers.CumSumLayer(lddia3dropnorm)

    submodels = [meta_2ch, meta_4ch]
    return {
        "inputs": dict({
            "sliced:meta:PatientAge": l_age,
            "sliced:meta:PatientSex": l_sex,
        }, **{ k: v for d in [model["inputs"] for model in submodels]
               for k, v in d.items() }
        ),
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": dict({
        }, **{ k: v for d in [model["regularizable"] for model in submodels if "regularizable" in model]
               for k, v in d.items() }
        ),
        "pretrained":{
            j6_2ch_128mm.__name__: meta_2ch["outputs"],
            j6_4ch.__name__: meta_4ch["outputs"],
        },
        "cutoff_gradients": [
        ] + [ v for d in [model["meta_outputs"] for model in submodels if "meta_outputs" in model]
               for v in d.values() ]
    }

