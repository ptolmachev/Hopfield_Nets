import numpy as np
import pickle
from src.Hopfield_net import *
from matplotlib import pyplot as plt
from copy import deepcopy
from src.visualisation import flips_and_patterns_contour_plot

num_neurons = 100
num_of_flips = 100 - 1
num_of_patterns = 120
num_repetitions = 10
time = 5
num_it = 100
sync = True
incremental = True
lmbd = 0.5
rules = ['DescentL2', 'DescentOverlap']
options = dict()
options['num_it'] = num_it
options['lambda'] = lmbd
options['incremental'] = incremental
sc = True
symm = True
for rule in rules:
    # for symm in [True, False]:
    # for sc in [True, False]:
    options['rule'] = rule
    options['symm'] = symm
    options['sc'] = sc
    file_name = f'../data/flips_and_patterns_{rule}_sc={sc}_symm={symm}_incremental={incremental}_lmbd={lmbd}_{num_neurons}x{num_of_patterns}.pkl' #_
    results = np.zeros((num_of_patterns, num_of_flips, num_repetitions))

    HN = Hopfield_network(num_neurons=num_neurons)
    for r in range(num_repetitions):
        print(f'repetition number {r}')
        for n_p in range(1,num_of_patterns+1):
            patterns = [random_state(0.5, num_neurons) for i in range(n_p)]
            HN.set_weights(np.zeros((num_neurons, num_neurons)))
            HN.learn_patterns(patterns, options)
            for n_f in range(1,num_of_flips+1):
                print(f'Introducing {n_f} flips')
                pattern_r = deepcopy(introduce_random_flips(patterns[0], n_f))
                retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
                overlap = (1 / num_neurons) * np.dot(retrieved_pattern,patterns[0])
                results[n_p - 1, n_f - 1, r] = overlap
        if (n_p % 10 == 0):
            pickle.dump(results, open(file_name,'wb+'))
    # flips_and_patterns_contour_plot(file_name)
    pickle.dump(results, open(file_name,'wb+'))
