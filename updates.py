import lasagne
from configuration import config

def build_nesterov_updates(train_loss, all_params, learning_rate):
    updates = lasagne.updates.nesterov_momentum(train_loss, all_params, learning_rate, config().momentum)
    return updates

def build_adam_updates(train_loss, all_params, learning_rate):
    updates = lasagne.updates.adam(train_loss, all_params, learning_rate)
    return updates