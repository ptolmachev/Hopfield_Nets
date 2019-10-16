import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from learning_rules import *
from math_utils import random_state, introduce_random_flips, sigmoid
################# HOPFIELD NETWORK #################

class Hopfield_network():
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        Y = np.random.randn(self.num_neurons, self.num_neurons)
        R = (Y + Y.T) / 2
        R[np.arange(num_neurons), np.arange(num_neurons)] = np.zeros(num_neurons)
        self.weights = (1 / self.num_neurons) * R
        # self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.biases = np.zeros(self.num_neurons)
        self.hidden_state = np.random.rand(num_neurons)-0.5
        self.state = np.sign(self.hidden_state)

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = (self.weights @ self.state.reshape(-1, 1) + self.biases.reshape(-1, 1)).flatten()
            self.state = np.sign(self.hidden_state)
        elif sync == False:
            for i in np.random.permutation(list(range(self.num_neurons))):
                self.hidden_state[i] = self.weights[i, :].reshape(1, -1) @ self.state.reshape(-1, 1) + self.biases[i]
                self.state[i] = np.sign(self.hidden_state[i])
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def learn_patterns(self, patterns, rule, options):
        patterns = deepcopy(np.array(patterns).reshape(-1, len(patterns[0])))
        if rule == 'Hebb':
            self.weights, self.biases = hebbian_lr(self.num_neurons, patterns, self.weights, self.biases,  **options)
        elif rule == 'Pseudoinverse':
            self.weights, self.biases = pseudoinverse(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'Storkey':
            self.weights, self.biases = storkey(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'StorkeyNormalisedLF':
            self.weights, self.biases = storkey_normalised_lf(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'DescentL2Solver':
            self.weights, self.biases = descent_l2_with_solver(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'DescentL1Solver':
            self.weights, self.biases = descent_l1_with_solver(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'DescentCESolver':
            self.weights, self.biases = descent_crossentropy_with_solver(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'DescentAnalyticalCentre':
            self.weights, self.biases = descent_analytical_centre_with_solver(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'L2DifferenceMin':
            self.weights, self.biases = l2_difference_minimisation(self.num_neurons, patterns, self.weights, self.biases, **options)
        else:
            raise ValueError('the specified learning rule is not implemented')
        return None

    def set_params(self, new_weights, new_biases):
        self.weights = new_weights
        self.biases = new_biases
        return None

    def retrieve_pattern(self, initial_state, sync, time, record = False):
        # global data
        self.state = deepcopy(initial_state)
        t = 0
        if record == True:
            data = dict()
            data['hidden_variables'] = []
            data['state'] = []
        while t < time:
            self.update_state(sync)
            t += 1
            if record == True:
                data['hidden_variables'].append(deepcopy(self.hidden_state))
                data['state'].append(deepcopy(self.state))

        if record == True:
            data['hidden_variables'] = deepcopy(np.array(data['hidden_variables']).T)
            data['state'] = deepcopy(np.array(data['state']).T)
            return self.state, data
        else:
            return self.state



if __name__ == '__main__':
    num_neurons = 100
    num_patterns = 30
    sync = True
    flips = 20
    time = 50
    num = 2
    rule = 'DescentAnalyticalCentre'
    options = {'incremental' : True, 'tol' : 1e-3, 'lmbd' : 0.5, 'alpha' : 0.001}
    HN = Hopfield_network(num_neurons=num_neurons)
    patterns = [random_state(p=0.5, n=num_neurons, values=[-1, 1]) for i in range(num_patterns)]
    HN.learn_patterns(patterns, rule, options)
    pattern_r = introduce_random_flips(pattern=patterns[num], k=flips, values=[-1, 1])

    print('\nSimilarity with different patterns (before): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(pattern_r - patterns[i], 2))

    retrieved_pattern, data = HN.retrieve_pattern(pattern_r, sync, time, record = True)
    print('\nSimilarity with different patterns (after): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(retrieved_pattern - patterns[i], 2))

    fig1 = plt.figure()
    plt.plot(data['hidden_variables'].T)
    plt.show()
    fig3 = plt.figure()
    similarities = np.array([[np.dot(data['state'][:,j],patterns[i])/num_neurons for i in range(len(patterns))] for j in range(time)])
    plt.plot(similarities)
    plt.show()
    # print('\nSimilarity of signs with different patterns (after): ')
    # for i in range(len(patterns)):
    #     print(np.linalg.norm(np.sign(retrieved_pattern) - patterns[i], 2))