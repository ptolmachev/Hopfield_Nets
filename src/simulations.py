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
                patterns = deepcopy([random_state(0.5, num_neurons) for i in range(n_p)])

                R = np.random.randn(num_neurons, num_neurons)
                R = (R + R.T) / 2
                R[np.arange(num_neurons), np.arange(num_neurons)] = np.zeros(num_neurons)

                HN.set_params((1 / num_neurons) * R, np.zeros(num_neurons))
                if ('unlearning' in options[i].keys()) and (options[i]['unlearning'] == True):
                    options[i]['HN'] = HN

                HN.learn_patterns(patterns, rule, options[i])
                for n_f in range(1, num_of_flips+1):
                    num = np.random.randint(len(patterns))
                    true_pattern = deepcopy(patterns[num])
                    pattern_r = deepcopy(introduce_random_flips(true_pattern, n_f))
                    retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
                    overlap = (1 / num_neurons) * np.dot(retrieved_pattern, true_pattern)
                    results[n_p - 1, n_f - 1, r] = overlap
        pickle.dump(results, open(file_name,'wb+'))
    return None

if __name__ == '__main__':
    params = dict()
    params['rules'] = ['PseudoinverseNondiag']
    params['options'] = [{}]
    #{'activation_function' : 'linear', 'tol' : 1e-1, 'lmbd' : 0.5}
    # params['options'] = [{'sc': True, 'incremental' : True, 'unlearning' : True, 'HN' : True, 'unlearn_rate' : 0.1, 'num_of_retrieval' : 10, 'sync' : True, 'time' : 10}]
    params['time_of_retrieval'] = 50
    params['sync'] = True
    num_neurons = 50
    num_of_flips = 50 - 1
    num_of_patterns = 75
    num_repetitions = 100
    flips_and_patterns(num_neurons, num_of_flips, num_of_patterns, num_repetitions, params)