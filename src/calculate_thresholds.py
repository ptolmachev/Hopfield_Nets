import pickle
import matplotlib
from copy import deepcopy
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
    # avg = pd.DataFrame(np.mean(results, axis=-1)).fillna(0).values
    boundaries = []
    std_err = []
    for k in range(results.shape[-1]):
        boundary = []
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                if (results[i,0,k] < epsilon):
                    boundary.append(0)
                    break
                elif (results[i,j, k] > epsilon) and (np.mean(results[i,j:j+5, k]) < epsilon) :
                    boundary.append(j)
                    break
                elif j == (results.shape[1] - 1):
                    boundary.append(j)
                    break
                else:
                    pass
        boundaries.append(deepcopy(boundary))
    boundary_file_name = file_name.split('.pkl')[0] + '_boundary.pkl'
    data = dict()
    data['boundary'] = np.mean(np.array(boundaries), axis=0)
    data['std'] = np.std(np.array(boundaries), axis=0)
    pickle.dump(data, open(boundary_file_name,'wb+'))
    return None

def generate_comparison_plot(rules, options, incremental):
    fig = plt.figure(figsize=(7,10))
    colors = ['saddlebrown', 'm', 'g','b', 'g', 'r', 'purple', 'slategray', '#19a2f3','#8a6cb4']
    for i, rule in enumerate(rules):
        arguments = options[i]
        file_name = f'../data/flips_and_patterns_{get_postfix(rule, arguments, num_neurons, num_of_patterns, num_repetitions)}.pkl'
        boundary_file_name = file_name.split('.pkl')[0] + '_boundary.pkl'
        data = pickle.load(open(boundary_file_name, 'rb+'))
        boundary = data['boundary']
        std_err = data['std']
        if rule == 'Pseudoinverse':
            linestyle = '-'
            color = 'k'
            linewidth = 2
        else:
            linestyle = '-'
            color = colors[i]
            linewidth = 2

        # plt.fill_between(x, y1, y2)
        y0 = savgol_filter(boundary, 9, 3)
        n = len(y0)
        x = np.arange(n)
        y1 = savgol_filter(np.array(boundary) - 0.25*np.array(std_err), 9, 3)
        y2 = savgol_filter(np.array(boundary) + 0.25*np.array(std_err), 9, 3)

        plt.plot(y0, x, color=color,
                 label=rule, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
        plt.plot(y1, x, color=color, linestyle=linestyle, linewidth=0.5, alpha=0.2)
        plt.plot(y2, x, color=color, linestyle=linestyle, linewidth=0.5, alpha=0.2)
        plt.fill_betweenx(x, y1, y2, color=color, alpha = 0.02)
        plt.legend(loc='upper right', shadow=True, fontsize='x-large')
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
    num_neurons = 75
    num_of_patterns = 75
    num_repetitions = 100

    # rules_incremental = ['Hebbian', 'Storkey', 'KrauthMezard', 'DiederichOpperI', 'DiederichOpperII',
    #          'DescentBarrier', 'DescentL1', 'DescentL2', 'DescentCE',
    #          'DescentBarrierNormalisedOverlap', 'Gardner']
    # options_incremental = [ {'incremental' : True, 'sc' : True }, #Hebbian
    #             {'incremental': True, 'sc': True, 'order' : 2}, #Storkey
    #             {'lr': 1e-2, 'max_iter' : 200}, #Krauth-Mezard
    #             {'lr': 1e-2}, #DOI
    #             {'lr': 1e-2, 'tol': 1e-2}, #DOII
    #             {'incremental' : True, 'tol' : 1e-2, 'lmbd' : 0.5, 'alpha' : 0.01}, #DescentBarrier
    #             {'incremental' : True, 'tol' : 1e-2, 'lmbd' : 0.5, 'alpha' : 0.01}, #DescentL1
    #             {'incremental' : True, 'tol' : 1e-2, 'lmbd' : 0.5, 'alpha' : 0.01}, #DescentL2
    #             {'incremental' : True, 'tol' : 1e-2, 'lmbd' : 0.5, 'alpha' : 0.01}, #DescentCE
    #             {'incremental' : True, 'tol' : 1e-2, 'lmbd' : 0.5}, #DescentNormalisedOverlap
    #             {'lr' :  1e-2, 'k' : 1.0, 'max_iter' : 200} #Gardner
    #             ]


    rules_nonincremental = ['Hebb', 'Storkey', 'Pseudoinverse', 'KrauthMezard',
             'DescentBarrier', 'DescentBarrierNormalisedOverlap', 'DescentL1', 'DescentL2']#, 'DescentCE' ]
    options_nonincremental = [{'incremental' : False, 'sc' : True },  #Hebbian
               {'incremental' : False, 'sc': True, 'order': 2},  # Storkey
               {},  #Pseudoinverse
               {'lr': 1e-2, 'max_iter': 200},  # Krauth-Mezard
               {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.01},  #DescentBarrier
               {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5},  # DescentNormalisedOverlap
               {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.01},  #DescentL1
               {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.01},  #DescentL2
               # {'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.01}  #DescentCE
               ]

    # for i in range(len(rules_incremental)):
    #     rule = rules_incremental[i]
    #     arguments = options_incremental[i]
    #     file_name = f'../data/flips_and_patterns_{get_postfix(rule, arguments, num_neurons, num_of_patterns, num_repetitions)}.pkl'
    #     get_bound(file_name, epsilon)
    # generate_comparison_plot(rules_incremental, options_incremental, True)

    for i in range(len(rules_nonincremental)):
        rule = rules_nonincremental[i]
        arguments = options_nonincremental[i]
        file_name = f'../data/flips_and_patterns_{get_postfix(rule, arguments, num_neurons, num_of_patterns, num_repetitions)}.pkl'
        get_bound(file_name, epsilon)
    generate_comparison_plot(rules_nonincremental, options_nonincremental, False)

