
import numpy as np
import pickle
from src.Hopfield_net import *
from matplotlib import pyplot as plt

from src.visualisation import flips_and_patterns_contour_plot

num_neurons = 40
num_of_flips = 40
num_of_patterns = 40
num_repetitions = 10
sync = True
time = 50
rule = 'Optimised'
sc = True
incremental = True
sc_or_not = '' if sc == False else '_sc'
file_name = f'../data/flips_and_patterns_{rule}{sc_or_not}_{num_neurons}.pkl'
results = np.zeros((num_of_patterns, num_of_flips,num_repetitions))
for n_p in range(1,num_of_patterns+1):
    print(f'\n Testing memorising {n_p} patterns')
    for n_f in range(1,num_of_flips+1):
        print(f'Introducing {n_f} flips')
        for r in range(num_repetitions):
            HN = Hopfield_network(num_neurons=num_neurons)
            patterns = [random_state(0.5, num_neurons) for i in range(n_p)]
            HN.learn_patterns(patterns, rule=rule, sc=sc, incremental=incremental)
            pattern_r = introduce_random_flips(patterns[0], n_f)
            retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
            overlap = (1/num_neurons)*np.dot(retrieved_pattern,patterns[0])
            # print(overlap)
            results[n_p-1,n_f-1,r] = overlap
    if (n_p % 10 == 0):
        pickle.dump(results, open(file_name,'wb+'))

flips_and_patterns_contour_plot(file_name)
