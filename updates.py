import lasagne
from configuration import config

def build_updates(train_loss, all_params, learning_rate):
    updates = lasagne.updates.nesterov_momentum(train_loss, all_params, learning_rate, config().momentum)
    return updates