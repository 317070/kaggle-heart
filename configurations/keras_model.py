from collections import namedtuple
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import data
import data_iterators
import nn_heart

patch_size = (64, 64)
augmentation_params = {
    'zoom_range': (1 / 1.2, 1.2),
    'rotation_range': (0, 360),
    'translation_range': (-8, 8),
    'do_flip': True
}

batch_size = 128
num_batches_train = 500

momentum = 0.9
learning_rate_schedule = {}

validate_every = 20
save_every = 20

data_iterator = data_iterators.DataLoader(num_batches_train=num_batches_train,
                                          patch_size=patch_size, batch_size=batch_size,
                                          augmentation_params=augmentation_params)


def build_model():
    l_outs = []
    top_layer = nn.layers.MergeLayer(incomings=l_outs)

    return namedtuple('Model', ['l_ins', 'l_outs', 'l_targets' 'l_top'])([], [], [], None)


def build_objective(model):
    predictions = [nn.layers.get_output(l) for l in model.l_outs]
    targets = [nn.layers.get_output(l) for l in model.l_targets]
    predictions_det = [nn.layers.get_output(l, deterministic=True) for l in model.l_outs]


#    T.mean((cdf_predictions - cdf_targets) ** 2, axis=[0, 1])

def build_updates(train_loss, model, learning_rate):
    pass
