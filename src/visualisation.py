# import sys
# sys.path.insert(0, '../src')
from Hopfield_net import Hopfield_network, random_state, introduce_random_flips
import unittest
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from copy import deepcopy
import pickle
import pandas as pd


def visualise_patterns(patterns, initial_state, retrieved_pattern):
    n = (int(np.sqrt(initial_state.shape[-1])))
    fig = plt.figure()
    gs = fig.add_gridspec(3, 4)
    axs = [fig.add_subplot(gs[0, i]) for i in range(len(patterns))]
    axs.append(fig.add_subplot(gs[1:3, 0:2]))
    axs.append(fig.add_subplot(gs[1:3, 2:4]))
    for i, pattern in enumerate(patterns):
        patterns[i] = pattern.reshape( (n,n) )
        axs[i].set_title(f'pattern {i+1}')
        axs[i].set_axis_off()
        axs[i].imshow(patterns[i],)

    axs[4].set_title(f'Initial State')
    axs[4].imshow(initial_state.reshape( (n,n) ) )
    axs[4].set_axis_off()
    axs[5].set_title(f'Retrieved Pattern')
    axs[5].imshow(retrieved_pattern.reshape( (n,n) ) )
    axs[5].set_axis_off()
    plt.tight_layout()
    plt.show()


def run_visualisation(rule = 'pseudoinverse', incremental = False, flips = 10, sync = False, time = 50, random_patterns = False):
    n = 10
    num_neurons = n**2
    HN = Hopfield_network(num_neurons=num_neurons)
    patterns = [random_state(0.7,n**2) for i in range(4)]
    smile_positive = -np.ones((n,n))
    smile_positive[2,3] = smile_positive[2,6] = smile_positive[3,3] = smile_positive[3,6] = smile_positive[4,3] = smile_positive[4,6] = 1
    smile_positive[6,1] = smile_positive[7,2] = smile_positive[6,8] = smile_positive[7,7] = 1
    smile_positive[8,3] = smile_positive[8,4] = smile_positive[8,5] = smile_positive[8,6]  = 1

    smile_negative = -np.ones((n,n))
    smile_negative[2,3] = smile_negative[2,6] = smile_negative[3,3] = smile_negative[3,6] = smile_negative[4,3] = smile_negative[4,6] = 1
    smile_negative[8,1] = smile_negative[7,2] = smile_negative[7,7] = smile_negative[8,8] = 1
    smile_negative[6,3] = smile_negative[6,4] = smile_negative[6,5] = smile_negative[6,6] = 1

    letter_T = -np.ones((n,n))
    letter_T[0:2,:] *= -1
    letter_T[2:,4:6] *= -1

    letter_E = -np.ones((n,n))
    letter_E[0:2,:] *= -1
    letter_E[4:6,:] *= -1
    letter_E[8:,:] *= -1
    letter_E[:,0:2] = 1
    if random_patterns == False:
        patterns[0] = smile_positive.flatten()
        patterns[1] = smile_negative.flatten()
        patterns[2] = letter_T.flatten()
        patterns[3] = letter_E.flatten()

    HN.learn_patterns(patterns, rule=rule, incremental=incremental)
    pattern_r = introduce_random_flips(patterns[0], flips)
    retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
    visualise_patterns(patterns, pattern_r, retrieved_pattern)
    return None


def flips_and_patterns_contour_plot(file_name):
    results = pickle.load(open(file_name, 'rb+'))
    sc = True if '_sc_' in file_name else False
    rule = file_name.split('.pkl')[0].split('_')[-3 if sc else -2]
    # results = (results[:, :results.shape[1] // 2, :] - results[:, results.shape[1] // 2:, :][:, ::-1, :]) / 2
    avg = pd.DataFrame(np.mean(results, axis=-1)).fillna(0)

    fig, axs = plt.subplots(1, 1)
    cs = axs.contourf(avg, levels=np.linspace(-1., 1.0, 41), cmap='coolwarm', extend='both', linestyles='solid')
    axs.contour(avg, levels=np.linspace(-1, 1, 41), colors='k', linestyles='dashed', linewidths=0.25)
    axs.set_xlabel('Number of flips in the initial state', fontsize=24)
    axs.set_ylabel('Number of memorised patterns', fontsize=24)
    # axs.set_xticklabels(20*np.arange(5), fontsize=16)
    # axs.set_yticklabels(20*np.arange(5), fontsize=16)
    axs.set_title(
        f'The dependence of overlap between the true and retrieved states on flips \n in initial conditions and number of stored patterns (rule = {rule}, sc = {sc})',
        fontsize=24)
    fig.colorbar(cs)
    plt.show()
    fig.savefig('../imgs/' + file_name.split('.pkl')[0].split('/')[-1] + '.png')

if __name__ == '__main__':
    flips_and_patterns_contour_plot('../data/flips_and_patterns_StorkeyAsymm_150.pkl')