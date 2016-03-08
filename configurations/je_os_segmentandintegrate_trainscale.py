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
batches_per_chunk = 8
num_epochs_train = 400

# - learning rate and method
base_lr = 0.0003
learning_rate_schedule = {
    0: base_lr,
    9*num_epochs_train/10: base_lr/10,
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
    "rotation": (-360, 360),
    "shear": (0, 0),
    "translation": (-8, 8),
    "flip_vert": (0, 1),
    "roll_time": (0, 0),
    "flip_time": (0, 0),
}

patch_mm = 64
use_hough_roi = True
preprocess_train = functools.partial(  # normscale_resize_and_augment has a bug
    preprocess.preprocess_normscale,
    normscale_resize_and_augment_function=functools.partial(
        image_transform.normscale_resize_and_augment_2,
        normalised_patch_size=(patch_mm, patch_mm)))
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
    # don't use patients who don't have more than 6 slices
    return [
        folder for folder in folders
        if data_loader.compute_nr_slices(folder) > 6]

# Input sizes
patch_px = 32
nr_slices = 22
data_sizes = {
    "sliced:data:sax": (batch_size, nr_slices, 30, patch_px, patch_px),
    "sliced:data:sax:locations": (batch_size, nr_slices),
    "sliced:data:sax:is_not_padded": (batch_size, nr_slices),
    "sliced:data:randomslices": (batch_size, nr_slices, 30, patch_px, patch_px),
    "sliced:data:singleslice:difference:middle": (batch_size, 29, patch_px, patch_px),
    "sliced:data:singleslice:difference": (batch_size, 29, patch_px, patch_px),
    "sliced:data:singleslice": (batch_size, 30, patch_px, patch_px),
    "sliced:data:ax": (batch_size, 30, 15, patch_px, patch_px),
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, patch_px, patch_px)
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
    l0_slices = nn.layers.ReshapeLayer(l0, (batch_size * nr_slices, 30, patch_px, patch_px)) # (bxs, t, i, j)
    subsample_factor = 2
    l0_slices_subsampled = nn.layers.SliceLayer(l0_slices, axis=1, indices=slice(0, 30, subsample_factor))
    nr_frames_subsampled = 30 / subsample_factor

    # PREPROCESS FRAMES SEPERATELY
    l0_frames = nn.layers.ReshapeLayer(l0_slices_subsampled, (batch_size * nr_slices * nr_frames_subsampled, 1, patch_px, patch_px))  # (bxsxt, 1, i, j)

    # downsample
    downsample = lambda incoming: nn.layers.dnn.Pool2DDNNLayer(incoming, pool_size=(2,2), stride=(2,2), mode='average_inc_pad')
    upsample = lambda incoming: nn.layers.Upscale2DLayer(incoming, scale_factor=2)
    l0_frames_d0 = l0_frames
    l0_frames_d1 = downsample(l0_frames_d0)
    l0_frames_d2 = downsample(l0_frames_d1)
    l0_frames_d3 = downsample(l0_frames_d2)

    ld3a = nn.layers.dnn.Conv2DDNNLayer(l0_frames_d3,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=16, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld3b = nn.layers.dnn.Conv2DDNNLayer(ld3a,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=16, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld3c = nn.layers.dnn.Conv2DDNNLayer(ld3b,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=16, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld3o = nn.layers.dnn.Conv2DDNNLayer(ld3c,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=16, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)

    ld2i = nn.layers.ConcatLayer([l0_frames_d2, upsample(ld3o)], axis=1)
    ld2a = nn.layers.dnn.Conv2DDNNLayer(ld2i,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld2b = nn.layers.dnn.Conv2DDNNLayer(ld2a,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld2c = nn.layers.dnn.Conv2DDNNLayer(ld2b,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld2d = nn.layers.dnn.Conv2DDNNLayer(ld2c,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld2o = nn.layers.dnn.Conv2DDNNLayer(ld2d,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)

    ld1i = nn.layers.ConcatLayer([l0_frames_d1, upsample(ld2o)], axis=1)
    ld1a = nn.layers.dnn.Conv2DDNNLayer(ld1i,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld1b = nn.layers.dnn.Conv2DDNNLayer(ld1a,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld1c = nn.layers.dnn.Conv2DDNNLayer(ld1b,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld1d = nn.layers.dnn.Conv2DDNNLayer(ld1c,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld1o = nn.layers.dnn.Conv2DDNNLayer(ld1d,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)

    ld0i = nn.layers.ConcatLayer([l0_frames_d0, upsample(ld1o)], axis=1)
    ld0a = nn.layers.dnn.Conv2DDNNLayer(ld0i,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld0b = nn.layers.dnn.Conv2DDNNLayer(ld0a,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld0c = nn.layers.dnn.Conv2DDNNLayer(ld0b,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld0d = nn.layers.dnn.Conv2DDNNLayer(ld0c,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=32, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    ld0o = nn.layers.dnn.Conv2DDNNLayer(ld0d,  W=nn.init.Orthogonal("relu"), filter_size=(5,5), num_filters=1, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.sigmoid)
    ld0r = nn.layers.ReshapeLayer(ld0o, (batch_size * nr_slices * nr_frames_subsampled, patch_px, patch_px))

    l_frames_musigma = layers.IntegrateAreaLayer(ld0r, sigma_mode='scale', sigma_scale=.1)
    area_per_pixel_cm = (float(patch_mm) / float(patch_px))**2 / 100.0
    l_frames_musigma_cm = layers.TrainableScaleLayer(l_frames_musigma, scale=nn.init.Constant(area_per_pixel_cm), trainable=False)

    # Go back to a per slice model
    l_slices_musigma_cm = nn.layers.ReshapeLayer(l_frames_musigma_cm, (batch_size * nr_slices, nr_frames_subsampled, 2))  # (bxs, t, 2)
    l_slices_musigma_cm_sys = layers.ArgmaxAndMaxLayer(l_slices_musigma_cm, mode='min')  # (bxs, 2)
    l_slices_musigma_cm_dia = layers.ArgmaxAndMaxLayer(l_slices_musigma_cm, mode='max')  # (bxs, 2)
    l_slices_musigma_cm_avg = layers.ArgmaxAndMaxLayer(l_slices_musigma_cm, mode='mean')

    # AGGREGATE SLICES PER PATIENT
    l_scaled_slice_locations = layers.TrainableScaleLayer(lin_slice_locations, scale=nn.init.Constant(0.1), trainable=False)

    # Systole
    l_pat_sys_ss_musigma_cm = nn.layers.ReshapeLayer(l_slices_musigma_cm_sys, (batch_size, nr_slices, 2))
    l_pat_sys_ss_mu_cm = nn.layers.SliceLayer(l_pat_sys_ss_musigma_cm, indices=0, axis=-1)
    l_pat_sys_ss_sigma_cm = nn.layers.SliceLayer(l_pat_sys_ss_musigma_cm, indices=1, axis=-1)
    l_pat_sys_aggr_mu_sigma = layers.JeroenLayer([l_pat_sys_ss_mu_cm, l_pat_sys_ss_sigma_cm, lin_slice_mask, l_scaled_slice_locations], rescale_input=1.)

    l_systole = layers.MuSigmaErfLayer(l_pat_sys_aggr_mu_sigma)

    # Diastole
    l_pat_dia_ss_musigma_cm = nn.layers.ReshapeLayer(l_slices_musigma_cm_dia, (batch_size, nr_slices, 2))
    l_pat_dia_ss_mu_cm = nn.layers.SliceLayer(l_pat_dia_ss_musigma_cm, indices=0, axis=-1)
    l_pat_dia_ss_sigma_cm = nn.layers.SliceLayer(l_pat_dia_ss_musigma_cm, indices=1, axis=-1)
    l_pat_dia_aggr_mu_sigma = layers.JeroenLayer([l_pat_dia_ss_mu_cm, l_pat_dia_ss_sigma_cm, lin_slice_mask, l_scaled_slice_locations], rescale_input=1.)

    l_diastole = layers.MuSigmaErfLayer(l_pat_dia_aggr_mu_sigma)

    # Average
    l_pat_avg_ss_musigma_cm = nn.layers.ReshapeLayer(l_slices_musigma_cm_avg, (batch_size, nr_slices, 2))
    l_pat_avg_ss_mu_cm = nn.layers.SliceLayer(l_pat_avg_ss_musigma_cm, indices=0, axis=-1)
    l_pat_avg_ss_sigma_cm = nn.layers.SliceLayer(l_pat_avg_ss_musigma_cm, indices=1, axis=-1)
    l_pat_avg_aggr_mu_sigma = layers.JeroenLayer([l_pat_avg_ss_mu_cm, l_pat_avg_ss_sigma_cm, lin_slice_mask, l_scaled_slice_locations], rescale_input=1.)

    l_mean = layers.MuSigmaErfLayer(l_pat_avg_aggr_mu_sigma)


    return {
        "inputs":{
            "sliced:data:sax": l0,
            "sliced:data:sax:is_not_padded": lin_slice_mask,
            "sliced:data:sax:locations": lin_slice_locations,
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
            "average": l_mean,
        },
        "regularizable": {
        },
    }

