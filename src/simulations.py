import numpy as np
import pickle
from Hopfield_Network import *
from utils import *
from matplotlib import pyplot as plt
from copy import deepcopy
from visualisation import flips_and_patterns_contour_plot
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def flips_and_patterns(num_neurons, num_of_flips, num_of_patterns, num_repetitions, params, plot = False):
    rule = params['rule']
    learning_options = params['learning_options']
    retrieval_options = params['retrieval_options']
    time = retrieval_options['time_of_retrieval']
    sync = retrieval_options['sync']
    file_name = f'../data/flips_and_patterns_{get_postfix(rule, learning_options, num_neurons, num_of_patterns, num_repetitions)}.pkl'
    results = np.zeros((num_of_patterns, num_of_flips, num_repetitions))
    for r in range(num_repetitions):
        HN = Hopfield_network(num_neurons=num_neurons)
        print(f'repetition number {r}')
        for n_p in range(1,num_of_patterns+1):
            print(f'learning {n_p} patterns')
            patterns = deepcopy([random_state(0.5, num_neurons, values=[-1, 1]) for i in range(n_p)])

            R = np.random.randn(num_neurons, num_neurons)
            R = (R + R.T) / 2
            R[np.arange(num_neurons), np.arange(num_neurons)] = np.zeros(num_neurons)

            HN.set_params((1 / num_neurons) * R, np.zeros(num_neurons))

            HN.learn_patterns(patterns, rule, learning_options)
            for n_f in range(1, num_of_flips+1):
                num = np.random.randint(len(patterns))
                true_pattern = deepcopy(patterns[num])
                pattern_r = deepcopy(introduce_random_flips(true_pattern, n_f, values = [-1, 1]))
                retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
                overlap = (1 / num_neurons) * np.dot(retrieved_pattern, true_pattern)
                results[n_p - 1, n_f - 1, r] = overlap
    pickle.dump(results, open(file_name,'wb+'))
    if plot == True:
        file_name = f'../data/flips_and_patterns_{get_postfix(rule, learning_options, num_neurons, num_of_patterns, num_repetitions)}.pkl'
        flips_and_patterns_contour_plot(file_name)
    return None

def weights_distribution_plot(num_neurons, num_of_patterns, params):
    rule = params['rule']
    options = params['learning_options']
    HN = Hopfield_network(num_neurons=num_neurons)
    patterns = deepcopy([random_state(0.5, num_neurons) for i in range(num_of_patterns)])
    HN.learn_patterns(patterns, rule, options)

    w = HN.weights.flatten()
    fig = plt.figure()
    _ = plt.hist(w, density=True, facecolor='g', alpha=0.75, bins='auto')  # arguments are passed to np.histogram
    plt.xlabel('Weight Value')
    plt.ylabel('Empirical Probability Density')
    plt.title(f'Weight distribution (N = {num_neurons}, Num patterns = {num_of_patterns}, rule = {rule})')
    plt.grid(True)
    # plt.xticks(np.arange(np.round(min(w),-1), np.round(max(w) + 1,-1), 5))
    plt.savefig(f'../imgs/{rule}/WeightDistr_(N = {num_neurons}_Num patterns = {num_of_patterns}_rule = {rule})')
    # plt.show()
    plt.close(fig=fig)
    return None

if __name__ == '__main__':
    num_neurons = 75
    num_of_flips = 37
    num_of_patterns = 75
    num_repetitions = 100


    rules = [#'Hebb',
            # 'Storkey',
            # 'Pseudoinverse',
            # 'KrauthMezard',
            # 'DescentExpBarrier',
            # 'DescentExpBarrierSI',
            # 'DescentL1',
            # 'DescentL2',
            'DescentL2SI'
            # 'Hebb',
            # 'Storkey'
            # 'DiederichOpperI',
            # 'DiederichOpperII',
            # 'DescentExpBarrier',
            # 'DescentExpBarrierSI']
            # 'DescentL1',
            # 'DescentL2',
            # 'Gardner'
    ]
    options = [#{'incremental' : False, 'sc' : True },  #Hebbian
               # {'incremental' : False, 'sc': True},  # Storkey
               # {},  #Pseudoinverse
               # {'lr': 1e-2, 'max_iter': 200},  # Krauth-Mezard
               # {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentExpBarrier
               # {'incremental' : False, 'tol' : 1e-3, 'lmbd': 0.5},  # DescentExpBarrierSI
               # {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL1
               # {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL2
                {'incremental': False, 'tol': 1e-3}  # DescentL2SI
               # {'incremental': True, 'sc': True},  # Hebbian
               # {'incremental': True, 'sc': True},  # Storkey
               # {'lr': 1e-2},  # DOI
               # {'lr': 1e-2, 'tol': 1e-1},  # DOII
               # {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.001},  # DescentExpBarrier
               # {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5},  # DescentExpBarrierSI
               # {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.001},  # DescentL1
               # {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.001},  # DescentL2
               # {'lr': 1e-2, 'k': 1.0, 'max_iter': 100}  # GardnerKrauthMezard
               ]
    for i, rule in enumerate(rules):
        print(rule)
        print('\n')
        params = dict()
        params['rule']  = rule
        params['learning_options'] = options[i]
        params['retrieval_options'] = {'time_of_retrieval' : 50, 'sync' : True}
        flips_and_patterns(num_neurons, num_of_flips, num_of_patterns, num_repetitions, params)
    # for i in range(1,150):
    #     print(i)
    #     weights_distribution_plot(100, i, params)