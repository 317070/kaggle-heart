import numpy as np
import lasagne

def sunny_objective(input_layers, output_layer):
    return lasagne.objectives.Objective(output_layer, loss_function=nn_plankton.log_loss)