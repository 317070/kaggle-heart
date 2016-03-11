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
num_epochs_train = 470

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


def filter_samples(folders):
    # don't use patients who don't have 4ch
    import glob
    import os
    import cPickle as pickle
    from paths import TEMP_FILES_PATH
    d = pickle.load(open(TEMP_FILES_PATH+"pkl_train_slice2roi.pkl"))
    d.update(pickle.load(open(TEMP_FILES_PATH+"pkl_validate_slice2roi.pkl")))

    c = pickle.load(open(TEMP_FILES_PATH+"pkl_train_metadata.pkl"))
    c.update(pickle.load(open(TEMP_FILES_PATH+"pkl_validate_metadata.pkl")))
    def has_4ch(f):
        ch_slices = glob.glob(f+"/4ch_*.pkl")
        if len(ch_slices) > 0:
            patient_id = str(data_loader._extract_id_from_path(ch_slices[0]))
            slice_name = os.path.basename(ch_slices[0])
            heart_size = max(float(d[patient_id][slice_name]['roi_radii'][0]) / c[patient_id][slice_name]['PixelSpacing'][0],
                             float(d[patient_id][slice_name]['roi_radii'][1]) / c[patient_id][slice_name]['PixelSpacing'][1])
            return (heart_size>=32)
        else:
            return False
    return [folder for folder in folders if has_4ch(folder)]



use_hough_roi = True  # use roi to center patches
preprocess_train = functools.partial(  # normscale_resize_and_augment has a bug
    preprocess.preprocess_normscale,
    normscale_resize_and_augment_function=functools.partial(
        image_transform.normscale_resize_and_augment_2, 
        normalised_patch_size=(256,256)))
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
test_time_augmentations = 1000  # More augmentations since a we only use single slices
tta_average_method = lambda x: np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))

# Architecture
def build_model():

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:singleslice:4ch"]

    l0 = nn.layers.InputLayer(input_size)

    l1a = nn.layers.dnn.Conv2DDNNLayer(l0,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1b = nn.layers.dnn.Conv2DDNNLayer(l1a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1 = nn.layers.dnn.MaxPool2DDNNLayer(l1b, pool_size=(2,2), stride=(2,2))

    l2a = nn.layers.dnn.Conv2DDNNLayer(l1,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2b = nn.layers.dnn.Conv2DDNNLayer(l2a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2 = nn.layers.dnn.MaxPool2DDNNLayer(l2b, pool_size=(2,2), stride=(2,2))

    l3a = nn.layers.dnn.Conv2DDNNLayer(l2,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3b = nn.layers.dnn.Conv2DDNNLayer(l3a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3c = nn.layers.dnn.Conv2DDNNLayer(l3b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3 = nn.layers.dnn.MaxPool2DDNNLayer(l3c, pool_size=(2,2), stride=(2,2))

    l4a = nn.layers.dnn.Conv2DDNNLayer(l3,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4b = nn.layers.dnn.Conv2DDNNLayer(l4a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4c = nn.layers.dnn.Conv2DDNNLayer(l4b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4 = nn.layers.dnn.MaxPool2DDNNLayer(l4c, pool_size=(2,2), stride=(2,2))

    l5a = nn.layers.dnn.Conv2DDNNLayer(l4,  W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5b = nn.layers.dnn.Conv2DDNNLayer(l5a, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5c = nn.layers.dnn.Conv2DDNNLayer(l5b, W=nn.init.Orthogonal("relu"), filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5 = nn.layers.dnn.MaxPool2DDNNLayer(l5c, pool_size=(2,2), stride=(2,2))

    # Systole Dense layers
    ldsys1 = nn.layers.DenseLayer(l5, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys1drop = nn.layers.dropout(ldsys1, p=0.5)
    ldsys2 = nn.layers.DenseLayer(ldsys1drop, num_units=512, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys2drop = nn.layers.dropout(ldsys2, p=0.5)
    ldsys3 = nn.layers.DenseLayer(ldsys2drop, num_units=600, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)

    ldsys3drop = nn.layers.dropout(ldsys3, p=0.5)  # dropout at the output might encourage adjacent neurons to correllate
    ldsys3dropnorm = layers.NormalisationLayer(ldsys3drop)
    l_systole = layers.CumSumLayer(ldsys3dropnorm)

    # Diastole Dense layers
    lddia1 = nn.layers.DenseLayer(l5, num_units=512, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia1drop = nn.layers.dropout(lddia1, p=0.5)
    lddia2 = nn.layers.DenseLayer(lddia1drop, num_units=512, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia2drop = nn.layers.dropout(lddia2, p=0.5)
    lddia3 = nn.layers.DenseLayer(lddia2drop, num_units=600, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.softmax)

    lddia3drop = nn.layers.dropout(lddia3, p=0.5)  # dropout at the output might encourage adjacent neurons to correllate
    lddia3dropnorm = layers.NormalisationLayer(lddia3drop)
    l_diastole = layers.CumSumLayer(lddia3dropnorm)


    return {
        "inputs":{
            "sliced:data:singleslice:4ch": l0
        },
        "outputs": {
            "systole": l_systole,
            "diastole": l_diastole,
        },
        "regularizable": {
            ldsys1: l2_weight,
            ldsys2: l2_weight,
            ldsys3: l2_weight_out,
            lddia1: l2_weight,
            lddia2: l2_weight,
            lddia3: l2_weight_out,
        },
    }

