import numpy as np

def get_postfix(rule, arguments, num_neurons, num_of_patterns, num_repetitions):
    postfix = rule
    for key in arguments.keys():
        if not(arguments[key] is None):
            postfix += '_' + key + '=' + str(arguments[key])
    return postfix + f'_{num_neurons}x{num_of_patterns}x{num_repetitions}'


def normalise_weights(weights):
    scale = np.sum(np.abs(weights), axis = 1)
    return weights / scale