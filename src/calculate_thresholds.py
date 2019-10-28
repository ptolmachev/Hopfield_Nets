import pickle
import matplotlib
from matplotlib import rc
rc('text', usetex=True)
# rc('font', family='serif')

from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from utils import *
import matplotlib.colors as mcolors


def get_bound(file_name, epsilon):
    results = pickle.load(open(file_name, 'rb+'))
    sc = True if 'sc=True' in file_name else False
    rule = file_name.split('.pkl')[0].split('_')[3]
    avg = pd.DataFrame(np.mean(results, axis=-1)).fillna(0).values
    # avg = (avg[:, :avg.shape[1]//2] - avg[:, avg.shape[1]//2 + 1:][:, ::-1])/2
    boundary = []
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            # if (avg[i,0] < epsilon):
            #     boundary.append(0)
            #     break
            if (avg[i,j] > epsilon) and (np.mean(avg[i,j:j+5]) < epsilon) :
                boundary.append(j)
                break
            if j == avg.shape[-1]:
                boundary.append(j)
    boundary_file_name = file_name.split('.pkl')[0] + '_boundary.pkl'
    pickle.dump(boundary, open(boundary_file_name,'wb+'))
    return None

def generate_comparison_plot(rules, options, incremental):
    fig = plt.figure(figsize=(7,10))
    colors = ['r', 'g', 'b', 'm', 'saddlebrown','indigo', 'purple', '#a6b9df','#19a2f3','#8a6cb4']
    for i, rule in enumerate(rules):
        arguments = options[i]
        file_name = f'../data/flips_and_patterns_{get_postfix(rule, arguments, num_neurons, num_of_patterns, num_repetitions)}.pkl'
        boundary_file_name = file_name.split('.pkl')[0] + '_boundary.pkl'
        boundary = pickle.load(open(boundary_file_name, 'rb+'))

        if rule == 'Pseudoinverse':
            linestyle = '-'
            color = 'k'
            linewidth = 3
        else:
            linestyle = '--'
            color = colors[i]
            linewidth = 2
        plt.plot(savgol_filter(boundary,5,3)[:90], np.arange(len(savgol_filter(boundary,5,3)[:90])), color=color, label=rule, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
        legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
        plt.grid(True)
    plt.minorticks_on()
    plt.ylabel('Number of patterns', fontsize = 20)
    plt.xlabel('Number of flips in an initial state', fontsize = 20)
    if incremental:
        plt.title('The threshold line for overlap = 0.95 between \n the retrieved and the intended state \n (incremental rules)',
                  fontsize = 27, y=0.996)
    else:
        plt.title(
            'The threshold line for overlap = 0.95 between \n the retrieved and the intended state \n (non-incremental rules)',
            fontsize=27, y=0.996)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize=20)
    suffix = 'incremental' if incremental == True else 'non-incremental'
    plt.savefig(f'../imgs/Comparison_{suffix}.png')
    plt.show()

    return None

if __name__ == '__main__':
    epsilon = 0.95
    num_neurons = 100
    num_of_patterns = 150
    num_repetitions = 100
    rules_incremental = ['KrauthMezard', 'DiederichOpperI', 'DiederichOpperII', 'DescentBarrier',
                         'DescentL1']#, 'DescentL2Newton', 'GardnerNewton', 'Storkey', 'Hebbian']
    options_incremental = [{'lmbd': 0.01, 'max_iter' : 100},
                           {'lmbd': 0.001},
                           {'lmbd': 0.01},
                           {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.01},
                           {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.01}]
                           # {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.01},
                           # {'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.01},
                           # {'incremental': True, 'sc' : True, 'order' : 2},
                           # {'incremental': True, 'sc': True}
                           # ]

    # rules_nonincremental = ['DescentBarrier', 'DescentCE', 'DescentL1', 'DescentL2', 'KrauthMezard', 'Pseudoinverse']
    #                         #, 'Hebbian', 'Storkey', ]
    # options_nonincremental = [{'incremental': False, 'tol': 1e-3, 'lmbd': 0.5, 'alpha': 0.001},
    #                           {'incremental': False, 'tol': 1e-3, 'lmbd': 0.5, 'alpha': 0.001},
    #                           {'incremental': False, 'tol': 1e-3, 'lmbd': 0.5, 'alpha': 0.001},
    #                           {'incremental': False, 'tol': 1e-3, 'lmbd': 0.5, 'alpha': 0.001},
    #                           {'lmbd': 0.01, 'max_iter': 100},
    #                           {}
    #                           ]
    #                        #    {'incremental': False, 'sc': True},
    #                        #    {'incremental': False, 'sc': True, 'order': 2},
    #                        #
    #                        # ]

    for i in range(len(rules_incremental)):
        rule = rules_incremental[i]
        arguments = options_incremental[i]
        file_name = f'../data/flips_and_patterns_{get_postfix(rule, arguments, num_neurons, num_of_patterns, num_repetitions)}.pkl'
        get_bound(file_name, epsilon)
    generate_comparison_plot(rules_incremental, options_incremental, True)

    # for i in range(len(rules_nonincremental)):
    #     rule = rules_nonincremental[i]
    #     arguments = options_nonincremental[i]
    #     file_name = f'../data/flips_and_patterns_{get_postfix(rule, arguments, num_neurons, num_of_patterns, num_repetitions)}.pkl'
    #     get_bound(file_name, epsilon)
    # generate_comparison_plot(rules_nonincremental, options_nonincremental, False)

