from default import *

import theano
import theano.tensor as T
import lasagne as nn

import deep_learning_layers
import preprocess
import postprocess
import objectives
import theano_printer
import updates

cached = "memory"

# Save and validation frequency
validate_every = 10
validate_train_set = False
save_every = 10
restart_from_save = False

dump_network_loaded_data = False

# Training (schedule) parameters
batch_size = 32
sunny_batch_size = 4
batches_per_chunk = 16
AV_SLICE_PER_PAT = 11
num_epochs_train = 150 * AV_SLICE_PER_PAT

learning_rate_schedule = {
    0:   0.00010,
    num_epochs_train/4:  0.00007,
    num_epochs_train/2:  0.00003,
    3*num_epochs_train/4: 0.00001,   
}

build_updates = updates.build_adam_updates


preprocess_train = preprocess.preprocess_with_augmentation
preprocess_validation = preprocess.preprocess  # no augmentation
preprocess_test = preprocess.preprocess_with_augmentation
test_time_augmentations = 100 * AV_SLICE_PER_PAT  # More augmentations since a we only use single slices
create_test_gen = partial(generate_test_batch, set='validation')  # validate as well by default

augmentation_params = {
    "rotation": (-16, 16),
    "shear": (0, 0),
    "translation": (-8, 8),
}

postprocess = postprocess.postprocess_value

# Input sizes
image_size = 128
data_sizes = {
    "sliced:data:singleslice:difference:middle": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice:difference": (batch_size, 29, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:singleslice": (batch_size, 30, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:ax": (batch_size, 30, 15, image_size, image_size), # 30 time steps, 30 mri_slices, 100 px wide, 100 px high,
    "sliced:data:shape": (batch_size, 2,),
    "sunny": (sunny_batch_size, 1, image_size, image_size)
    # TBC with the metadata
}


# Objective
l2_weight = 0.0005
def build_objective(interface_layers):
    # l2 regu on certain layers
    l2_penalty = nn.regularization.regularize_layer_params_weighted(
        interface_layers["regularizable"], nn.regularization.l2)
    # build objective
    return objectives.RMSEObjective(interface_layers["outputs"], penalty=l2_penalty)


# Architecture
def build_model():

    #################
    # Regular model #
    #################
    input_size = data_sizes["sliced:data:singleslice"]

    l0 = nn.layers.InputLayer(input_size)

    l1a = nn.layers.dnn.Conv2DDNNLayer(l0 , filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1b = nn.layers.dnn.Conv2DDNNLayer(l1a, filter_size=(3,3), num_filters=64, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l1 = nn.layers.dnn.MaxPool2DDNNLayer(l1b, pool_size=(2,2), stride=(2,2))

    l2a = nn.layers.dnn.Conv2DDNNLayer(l1 , filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2b = nn.layers.dnn.Conv2DDNNLayer(l2a, filter_size=(3,3), num_filters=128, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l2 = nn.layers.dnn.MaxPool2DDNNLayer(l2b, pool_size=(2,2), stride=(2,2))

    l3a = nn.layers.dnn.Conv2DDNNLayer(l2 , filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3b = nn.layers.dnn.Conv2DDNNLayer(l3a, filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3c = nn.layers.dnn.Conv2DDNNLayer(l3b, filter_size=(3,3), num_filters=256, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l3 = nn.layers.dnn.MaxPool2DDNNLayer(l3c, pool_size=(2,2), stride=(2,2))

    l4a = nn.layers.dnn.Conv2DDNNLayer(l3 , filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4b = nn.layers.dnn.Conv2DDNNLayer(l4a, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4c = nn.layers.dnn.Conv2DDNNLayer(l4b, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l4 = nn.layers.dnn.MaxPool2DDNNLayer(l4c, pool_size=(2,2), stride=(2,2))

    l5a = nn.layers.dnn.Conv2DDNNLayer(l4 , filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5b = nn.layers.dnn.Conv2DDNNLayer(l5a, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5c = nn.layers.dnn.Conv2DDNNLayer(l5b, filter_size=(3,3), num_filters=512, stride=(1,1), pad="same", nonlinearity=nn.nonlinearities.rectify)
    l5 = nn.layers.dnn.MaxPool2DDNNLayer(l5c, pool_size=(2,2), stride=(2,2))

    # Systole Dense layers
    ldsys1 = nn.layers.DenseLayer(l5, num_units=1024, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys1drop = nn.layers.dropout(ldsys1, p=0.5)
    ldsys2 = nn.layers.DenseLayer(ldsys1drop, num_units=1024, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    ldsys2drop = nn.layers.dropout(ldsys2, p=0.5)
    ldsys3 = nn.layers.DenseLayer(ldsys2drop, num_units=1, b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.identity)

    l_systole = deep_learning_layers.FixedScaleLayer(ldsys3, scale=1)

    # Diastole Dense layers
    lddia1 = nn.layers.DenseLayer(l5, num_units=1024, W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia1drop = nn.layers.dropout(lddia1, p=0.5)
    lddia2 = nn.layers.DenseLayer(lddia1drop, num_units=1024, W=nn.init.Orthogonal("relu"),b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.rectify)

    lddia2drop = nn.layers.dropout(lddia2, p=0.5)
    lddia3 = nn.layers.DenseLayer(lddia2drop, num_units=1, b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.identity)

    l_diastole = deep_learning_layers.FixedScaleLayer(lddia3, scale=1)


    return {
        "inputs":{
            "sliced:data:singleslice": l0
        },
        "outputs": {
            "systole:value": l_systole,
            "diastole:value": l_diastole,
            "systole:sigma": deep_learning_layers.FixedConstantLayer(np.ones((batch_size, 1), dtype='float32')*20./np.sqrt(test_time_augmentations)),
            "diastole:sigma": deep_learning_layers.FixedConstantLayer(np.ones((batch_size, 1), dtype='float32')*30./np.sqrt(test_time_augmentations)), 
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



