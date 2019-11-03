import pickle
import matplotlib
from copy import deepcopy
from matplotlib import rc
# rc('text', usetex=True)
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
    if incremental:
        colors = ['slateblue', 'b', 'g','m', 'olive', 'k', 'purple', 'darkorange', 'darkred','r']
    else:
        colors = ['slateblue', 'darkred', 'k', 'm', 'g', 'r', 'purple', 'darkorange', 'b', 'salmon']
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
            linewidth = 1.5
        else:
            linestyle = '-'
            color = colors[i]
            linewidth = 1.5

        # plt.fill_between(x, y1, y2)
        y0 = boundary #savgol_filter(boundary, 9, 3)
        n = len(y0)
        x = np.arange(n)
        y1 = np.array(boundary) - 0.25*np.array(std_err) #savgol_filter(np.array(boundary) - 0.25*np.array(std_err), 9, 3)
        y2 = np.array(boundary) + 0.25*np.array(std_err) #savgol_filter(np.array(boundary) + 0.25*np.array(std_err), 9, 3)

        y0 = savgol_filter(y0, 9, 3)
        y1 = savgol_filter(y1, 9, 3)
        y2 = savgol_filter(y2, 9, 3)

        plt.plot(y0, x, color=color,
                 label=rule, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
        plt.plot(y1, x, color=color, linestyle=linestyle, linewidth=0.5, alpha=0.2)
        plt.plot(y2, x, color=color, linestyle=linestyle, linewidth=0.5, alpha=0.2)
        plt.fill_betweenx(x, y1, y2, color=color, alpha = 0.02)
        plt.legend(loc='upper right', shadow=True, fontsize='x-large')
        plt.grid(True)
    plt.minorticks_on()
    plt.text(3, 3, 'Overlap > 0.95', fontsize=16)
    plt.text(17, 30, 'Overlap < 0.95', fontsize=16)
    plt.ylabel('Number of patterns', fontsize = 16)
    plt.xlabel('Number of flips in an initial state', fontsize = 16)
    if incremental:
        plt.title('The threshold line for overlap = 0.95 between \n the retrieved and the intended state \n (incremental rules)',
                  fontsize = 20, y=0.996)
    else:
        plt.title(
            'The threshold line for overlap = 0.95 between \n the retrieved and the intended state \n (non-incremental rules)',
            fontsize=20, y=0.996)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    suffix = 'incremental' if incremental == True else 'non-incremental'
    plt.savefig(f'../imgs/Comparison_{suffix}.png')
    plt.show()

    return None

if __name__ == '__main__':
    epsilon = 0.95
    num_neurons = 75
    num_of_patterns = 75
    num_repetitions = 100

    # rules_incremental = ['Hebb', 'Storkey', 'KrauthMezard', 'DiederichOpperI', 'DiederichOpperII',
    #          'DescentExpBarrier', 'DescentExpBarrierSI',
    #           'DescentL1', 'DescentL2',  'GardnerKrauthMezard'] #'DescentCE' ,
    # options_incremental = [ {'incremental' : True, 'sc' : True }, #Hebbian
    #             {'incremental': True, 'sc': True, 'order' : 2}, #Storkey
    #             {'lr': 1e-2, 'max_iter' : 200}, #Krauth-Mezard
    #             {'lr': 1e-2}, #DOI
    #             {'lr': 1e-2, 'tol': 1e-1}, #DOII
    #             {'sc' : True, 'incremental': True, 'tol': 1e-1, 'lmbd': 0.5, 'alpha': 0.001}, #DescentExpBarrier
    #             {'sc' : True,'incremental' : True, 'tol' : 1e-1, 'lmbd' : 0.5}, #DescentExpBarrierSI
    #             {'sc' : True,'incremental' : True, 'tol' : 1e-1, 'lmbd' : 0.5, 'alpha' : 0.001}, #DescentL1
    #             {'sc' : True,'incremental' : True, 'tol' : 1e-1, 'lmbd' : 0.5, 'alpha' : 0.001}, #DescentL2
    #             {'lr' :  1e-2, 'k' : 1.0, 'max_iter' : 75} #GardnerKrauthMezard
    #             ]


    rules_nonincremental = ['Hebb', 'Storkey', 'Pseudoinverse', 'KrauthMezard',
             'DescentExpBarrier', 'DescentExpBarrierSI', 'DescentL1', 'DescentL2', 'GardnerKrauthMezard']
    options_nonincremental = [
               {'incremental' : False, 'sc' : True },  #Hebbian
               {'incremental' : False, 'sc': True},  # Storkey
               {},  #Pseudoinverse
               {'sc' : True, 'lr': 1e-2, 'max_iter': 200},  # Krauth-Mezard
               {'sc' : True,'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentExpBarrier
               {'sc' : True,'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5},  # DescentExpBarrierSI
               {'sc' : True,'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL1
               {'sc' : True,'incremental' : False, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001},  #DescentL2
               {'sc' : True, 'lr' :  1e-2, 'k' : 1.0, 'max_iter' : 100}  #GardnerKrauthMezard
               ]

    rules_sc_comparison = ['Hebb', 'Storkey', 'DescentL2','GardnerKrauthMezard', 'DescentExpBarrierSI',
                           'Hebb', 'Storkey', 'DescentL2','GardnerKrauthMezard', 'DescentExpBarrierSI']
    options_sc_comparison = [
               {'incremental' : False, 'sc' : True },  #Hebbian
               {'incremental' : False, 'sc': True},  # Storkey
               {'sc' : True,'incremental': False, 'tol': 1e-3, 'lmbd': 0.5, 'alpha': 0.001},  # DescentL2
               {'sc' : True, 'lr' :  1e-2, 'k' : 1.0, 'max_iter' : 100},  #GardnerKrauthMezard
               {'sc' : True,'incremental' : True, 'tol' : 1e-1, 'lmbd' : 0.5}, #DescentExpBarrierSI
               {'incremental' : False, 'sc' : False},  #Hebbian
               {'incremental' : False, 'sc': False},  # Storkey
               {'sc': False, 'incremental': False, 'tol': 1e-3, 'lmbd': 0.5, 'alpha': 0.001},  # DescentL2
               {'sc': False, 'lr': 1e-2, 'k': 1.0, 'max_iter': 100},  # GardnerKrauthMezard
               {'sc': False, 'incremental': True, 'tol': 1e-1, 'lmbd': 0.5},  # DescentExpBarrierSI
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

