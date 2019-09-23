import numpy as np
import pickle
from src.Hopfield_net import *
from utils import *
from matplotlib import pyplot as plt
from copy import deepcopy
from src.visualisation import flips_and_patterns_contour_plot


def flips_and_patterns(num_neurons, num_of_flips, num_of_patterns, num_repetitions, params):
    rules = params['rules']
    options = params['options']
    time = params['time_of_retrieval']
    sync = params['sync']
    for i, rule in enumerate(rules):
        file_name = f'../data/flips_and_patterns_{get_postfix(rule, options[i], num_neurons, num_of_patterns)}.pkl'
        results = np.zeros((num_of_patterns, num_of_flips, num_repetitions))
        for r in range(num_repetitions):
            HN = Hopfield_network(num_neurons=num_neurons)
            print(f'repetition number {r}')
            for n_p in range(1,num_of_patterns+1):
                print(f'learning {n_p} patterns')
                patterns = [random_state(0.5, num_neurons) for i in range(n_p)]
                HN.set_params(np.random.randn(num_neurons, num_neurons), np.random.randn(num_neurons))
                HN.learn_patterns(patterns, rule, options[i])
                for n_f in range(1, num_of_flips+1):
                    pattern_r = deepcopy(introduce_random_flips(patterns[-1], n_f))
                    retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
                    overlap = (1 / num_neurons) * np.dot(retrieved_pattern, patterns[0])
                    results[n_p - 1, n_f - 1, r] = overlap
        pickle.dump(results, open(file_name,'wb+'))
    return None

if __name__ == '__main__':
    params = dict()
    params['rules'] = ['pseudoinverse']
    params['options'] = [{'sc' : True}]
    params['time_of_retrieval'] = 10
    params['sync'] = True
    num_neurons = 100
    num_of_flips = 100 - 1
    num_of_patterns = 150
    num_repetitions = 10
    flips_and_patterns(num_neurons, num_of_flips, num_of_patterns, num_repetitions, params)