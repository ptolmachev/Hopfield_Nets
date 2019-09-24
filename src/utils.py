import numpy as np

def get_postfix(rule, arguments, num_neurons, num_of_patterns):
    postfix = rule
    for key in arguments.keys():
        postfix += '_' + key + '=' + str(arguments[key])
    return postfix + f'_{num_neurons}x{num_of_patterns}'

def normalise_weights(weights):
    scale = np.sum(np.abs(weights), axis = 1)
    return weights / scale