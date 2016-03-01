from deep_learning_layers import ConvolutionOver2DAxisLayer, MaxPoolOverAxisLayer, MaxPoolOver2DAxisLayer, \
    MaxPoolOver3DAxisLayer, ConvolutionOver3DAxisLayer, ConvolutionOverAxisLayer
from default import *
import functools
import theano.tensor as T
from layers import MuLogSigmaErfLayer, CumSumLayer
import layers
import objectives
from lasagne.layers import InputLayer, reshape, DenseLayer, DenseLayer, batch_norm
from postprocess import upsample_segmentation
from volume_estimation_layers import GaussianApproximationVolumeLayer
import theano_printer
from updates import build_adam_updates
import image_transform

caching = None

validate_every = 10
validate_train_set = False
save_every = 10
restart_from_save = False

batches_per_chunk = 2

batch_size = 8
sunny_batch_size = 4
num_epochs_train = 60

image_size = 128

learning_rate_schedule = {
    0:     0.1,
    2:     0.01,
    10:    0.001,
    50:    0.0001,
    60:    0.00001,
}

from postprocess import postprocess_onehot, postprocess
from preprocess import preprocess, preprocess_with_augmentation, set_upside_up, normalize_contrast, preprocess_normscale, normalize_contrast_zmuv

use_hough_roi = True
preprocess_train = functools.partial(  # normscale_resize_and_augment has a bug
    preprocess_normscale,
    normscale_resize_and_augment_function=partial(
        image_transform.normscale_resize_and_augment_2, 
        normalised_patch_size=(80 ,80)))
#preprocess_train = preprocess_normscale
preprocess_validation = preprocess  # no augmentation
preprocess_test = preprocess_with_augmentation  # no augmentation
test_time_augmentations = 10
augmentation_params = {
    "rotate": (0, 0),
    "shear": (0, 0),
    "translate_x": (0, 0),
    "translate_y": (0, 0),
    "flip_vert": (0, 0),
    "zoom_x": (.75, 1.25),
    "zoom_y": (.75, 1.25),
    "change_brightness": (-0.3, 0.3),
}

cleaning_processes = [
    set_upside_up,]
cleaning_processes_post = [
    partial(normalize_contrast_zmuv, z=2)]

build_updates = build_adam_updates
postprocess = postprocess

nr_slices = 20
data_sizes = {
    "sliced:data:randomslices": (batch_size, nr_slices, 30, image_size, image_size),
    "sliced:data:sax:locations": (batch_size, nr_slices),
    "sliced:data:sax:is_not_padded": (batch_size, nr_slices),
    "sliced:data:sax": (batch_size, nr_slices, 30, image_size, image_size),
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 20 mri_slices, 100 px wide, 100 px high,
    "sliced:data:ax:noswitch": (batch_size, 15, 30, image_size, image_size), # 30 time steps, 20 mri_slices, 100 px wide, 100 px high,
    "area_per_pixel:sax": (batch_size, ),
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:middle": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}

check_inputs = False

def build_model():

    #################
    # Regular model #
    #################
    input_key = "sliced:data:singleslice:middle"
    data_size = data_sizes[input_key]

    l0 = InputLayer(data_size)
    l0r = batch_norm(reshape(l0, (-1, 1, ) + data_size[1:]))

    # (batch, channel, axis, time, x, y)

    # convolve over time
    l1 = batch_norm(ConvolutionOverAxisLayer(l0r, num_filters=8, filter_size=(3,), axis=(3,), channel=1,
                                   W=lasagne.init.Orthogonal(),
                                   b=lasagne.init.Constant(0.0),
                                   ))
    l1m = batch_norm(MaxPoolOverAxisLayer(l1, pool_size=(4,), axis=(3,)))

    # convolve over x and y
    l2a = batch_norm(ConvolutionOver2DAxisLayer(l1m, num_filters=8, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.0),
                                     ))
    l2b = batch_norm(ConvolutionOver2DAxisLayer(l2a, num_filters=8, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.0),
                                     ))
    l2m = batch_norm(MaxPoolOver2DAxisLayer(l2b, pool_size=(2, 2), axis=(4,5)))


    # convolve over x, y, time
    l3a = batch_norm(ConvolutionOver3DAxisLayer(l2m, num_filters=32, filter_size=(3, 3, 3),
                                     axis=(3,4,5), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     ))

    l3b = batch_norm(ConvolutionOver2DAxisLayer(l3a, num_filters=32, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     ))
    l3m = batch_norm(MaxPoolOver2DAxisLayer(l3b, pool_size=(2, 2), axis=(4,5)))

    # convolve over time
    l4 = batch_norm(ConvolutionOverAxisLayer(l3m, num_filters=32, filter_size=(3,), axis=(3,), channel=1,
                                   W=lasagne.init.Orthogonal(),
                                   b=lasagne.init.Constant(0.1),
                                   ))
    l4m = batch_norm(MaxPoolOverAxisLayer(l4, pool_size=(2,), axis=(2,)))

    # maxpool over axis
    l5 = batch_norm(MaxPoolOverAxisLayer(l3m, pool_size=(4,), axis=(2,)))

    # convolve over x and y
    l6a = batch_norm(ConvolutionOver2DAxisLayer(l5, num_filters=128, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     ))
    l6b = batch_norm(ConvolutionOver2DAxisLayer(l6a, num_filters=128, filter_size=(3, 3),
                                     axis=(4,5), channel=1,
                                     W=lasagne.init.Orthogonal(),
                                     b=lasagne.init.Constant(0.1),
                                     ))
    l6m = batch_norm(MaxPoolOver2DAxisLayer(l6b, pool_size=(2, 2), axis=(4,5)))

    # convolve over time and x,y, is sparse reduction layer
    l7 = ConvolutionOver3DAxisLayer(l6m, num_filters=32, filter_size=(3,3,3), axis=(3,4,5), channel=1,
                                   W=lasagne.init.Orthogonal(),
                                   b=lasagne.init.Constant(0.1),
                                   )

    key_scale = "area_per_pixel:sax"
    l_scale = InputLayer(data_sizes[key_scale])

    # Systole Dense layers
    ldsys1 = lasagne.layers.DenseLayer(l7, num_units=512,
                                  W=lasagne.init.Orthogonal("relu"),
                                  b=lasagne.init.Constant(0.1),
                                  nonlinearity=lasagne.nonlinearities.rectify)

    ldsys1drop = lasagne.layers.dropout(ldsys1, p=0.5)
    ldsys2 = lasagne.layers.DenseLayer(ldsys1drop, num_units=128,
                                       W=lasagne.init.Orthogonal("relu"),
                                       b=lasagne.init.Constant(0.1),
                                       nonlinearity=lasagne.nonlinearities.rectify)

    ldsys2drop = lasagne.layers.dropout(ldsys2, p=0.5)
    ldsys3 = lasagne.layers.DenseLayer(ldsys2drop, num_units=1,
                                       b=lasagne.init.Constant(0.1),
                                       nonlinearity=lasagne.nonlinearities.identity)

    l_systole = layers.MuConstantSigmaErfLayer(layers.ScaleLayer(ldsys3, scale=l_scale), sigma=0.0)

    # Diastole Dense layers
    lddia1 = lasagne.layers.DenseLayer(l7, num_units=512,
                                       W=lasagne.init.Orthogonal("relu"),
                                       b=lasagne.init.Constant(0.1),
                                       nonlinearity=lasagne.nonlinearities.rectify)

    lddia1drop = lasagne.layers.dropout(lddia1, p=0.5)
    lddia2 = lasagne.layers.DenseLayer(lddia1drop, num_units=128,
                                       W=lasagne.init.Orthogonal("relu"),
                                       b=lasagne.init.Constant(0.1),
                                       nonlinearity=lasagne.nonlinearities.rectify)

    lddia2drop = lasagne.layers.dropout(lddia2, p=0.5)
    lddia3 = lasagne.layers.DenseLayer(lddia2drop, num_units=1,
                                       b=lasagne.init.Constant(0.1),
                                       nonlinearity=lasagne.nonlinearities.identity)

    l_diastole = layers.MuConstantSigmaErfLayer(layers.ScaleLayer(lddia3, scale=l_scale), sigma=0.0)

    return {
        "inputs":{
            input_key: l0,
            key_scale: l_scale,
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": {
            ldsys1: l2_weight,
            ldsys2: l2_weight,
            ldsys3: l2_weight,
            lddia1: l2_weight,
            lddia2: l2_weight,
            lddia3: l2_weight,
        },
    }

l2_weight = 0.0005

def build_objective(interface_layers):
    # l2 regu on certain layers
    l2_penalty = lasagne.regularization.regularize_layer_params_weighted(interface_layers["regularizable"], lasagne.regularization.l2)
    # build objective
    return objectives.KaggleObjective(interface_layers["outputs"], penalty=l2_penalty)
