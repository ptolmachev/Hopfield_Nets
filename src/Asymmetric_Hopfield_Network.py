from math_utils import *
from Hopfield_Network import Hopfield_network
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr

################# HOPFIELD NETWORK WITH 0 AND 1 #################

class Asymmetric_Hopfield_network(Hopfield_network):
    def __init__(self, num_neurons):
        super().__init__(num_neurons)
        self.state = step(self.hidden_state)
        self.u = np.zeros(num_neurons)

    def update_state(self, sync=True):
        if sync == True:
            self.u += 0.00*(1.5*self.state - self.u)
            self.hidden_state = (self.weights @ self.state.reshape(-1, 1) + self.biases.reshape(-1, 1)).flatten() - self.u
            self.state = step(self.hidden_state)

        elif sync == False:
            for i in np.random.permutation(list(range(self.num_neurons))):
                self.hidden_state[i] = self.weights[i, :].reshape(1, -1) @ self.state.reshape(-1, 1) + self.biases[i]
                self.state[i] = step(self.hidden_state[i])
        else:
            raise AttributeError('\'sync\' variable can take only boolean values')
        return None

    def learn_patterns(self, patterns, rule, options):
        patterns = deepcopy(np.array(patterns).reshape(-1, len(patterns[0])))
        if rule == 'Hebb':
            self.weights, self.biases = hebbian_lr(self.num_neurons, patterns, self.weights, self.biases,  **options)
        elif rule == 'DescentL2Solver':
            self.weights, self.biases = descent_l2_with_solver_asymm(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'DescentL2SolverMulti':
            self.weights, self.biases = descent_l2_with_solver_multi(self.num_neurons, patterns, self.weights,
                                                                     self.biases, **options)
        elif rule == 'DescentL1Solver':
            self.weights, self.biases = descent_l1_with_solver_asymm(self.num_neurons, patterns, self.weights, self.biases, **options)
        elif rule == 'ChebyshevCentre':
            self.weights, self.biases = find_chebyshev_centre_asymm(self.num_neurons, patterns, self.weights, self.biases, **options)
        else:
            raise ValueError('the specified learning rule is not implemented')
        return None

    def retrieve_pattern(self, initial_state, sync, time, record = False):
        self.state = deepcopy(initial_state)
        t = 0
        if record == True:
            data = dict()
            data['hidden_variables'] = []
            data['state'] = []
            data['u'] = []

            data['hidden_variables'].append(deepcopy(self.hidden_state))
            data['state'].append(deepcopy(self.state))
            data['u'].append(deepcopy(self.u))
        while t < time:
            self.update_state(sync)
            t += 1
            if record == True:
                data['hidden_variables'].append(deepcopy(self.hidden_state))
                data['state'].append(deepcopy(self.state))
                data['u'].append(deepcopy(self.u))
        if record == True:
            data['hidden_variables'] = deepcopy(np.array(data['hidden_variables']).T)
            data['state'] = deepcopy(np.array(data['state']).T)
            data['u'] = deepcopy(np.array(data['u']).T)
            return self.state, data
        else:
            return self.state

#LEARNING RULES

def hebbian_lr(N, patterns, weights, biases, sc, incremental):
    weights = deepcopy(np.zeros((N, N)))
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            Y = ((2 * pattern - 1) @ (2 * pattern - 1).T)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        Y = (2 * Z - 1) @ (2 * Z - 1).T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights = Y
    return weights, biases

def descent_l2_with_solver_asymm(N, patterns, weights, biases, incremental, tol, lmbd):
    if incremental:
        for i in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[i].reshape(-1, 1)))
            x0 = np.concatenate([weights.flatten(), biases.flatten()])
            res = minimize(l2norm_difference_asymm, x0, args=(pattern, lmbd), jac = l2norm_difference_asymm_jacobian, method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
            biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    if incremental == False:
        Z = deepcopy(patterns).T.reshape(N, -1)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l2norm_difference_asymm, x0, args=(Z, lmbd), jac=l2norm_difference_asymm_jacobian,
                       method='L-BFGS-B', tol=tol, options={'disp': False})
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2:].reshape(-1, 1))
    return weights, biases

def descent_l2_with_solver_multi(N, patterns, weights, biases, incremental, tol, lmbd, mu, alpha):
    if incremental:
        for i in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[i].reshape(-1, 1)))
            x0 = np.concatenate([weights.flatten(), biases.flatten()])
            res = minimize(l2norm_difference_multi, x0, args=(pattern, lmbd, mu, alpha), jac = l2norm_difference_multi_jacobian, method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
            biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    if incremental == False:
        Z = deepcopy(patterns).T.reshape(N, -1)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l2norm_difference_multi, x0, args=(Z, lmbd, mu, alpha), jac=l2norm_difference_multi_jacobian,
                       method='L-BFGS-B', tol=tol, options={'disp': False})
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2:].reshape(-1, 1))
    return weights, biases

def descent_l1_with_solver_asymm(N, patterns, weights, biases, incremental, tol, lmbd):
    if incremental:
        for i in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[i].reshape(-1, 1)))
            x0 = np.concatenate([weights.flatten(), biases.flatten()])
            res = minimize(l1norm_difference_asymm, x0, args=(pattern, lmbd), jac = l1norm_difference_asymm_jacobian, method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
            biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    if incremental == False:
        Z = deepcopy(patterns).T.reshape(N, -1)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l1norm_difference_asymm, x0, args=(Z, lmbd), jac=l1norm_difference_asymm_jacobian,
                       method='L-BFGS-B', tol=tol, options={'disp': False})
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2:].reshape(-1, 1))
    return weights, biases

def find_chebyshev_centre_asymm(N, patterns, weights, biases, gamma):
    Z = deepcopy(patterns).T.reshape(N, -1)
    p = Z.shape[-1]
    W = np.array([])
    # separate the optimization task into subproblems for each neuron
    for i in range(N):
        A = -4*((Z - 1/2) * (Z[i, :] - 1/2)).T
        b = 0.0 * np.ones(p)
        x_c, R = chebyshev_centre(A, b, gamma)
        W = np.append(W, x_c)
    weights = W.reshape(N, N)
    return weights, biases


if __name__ == '__main__':
    num_neurons = 100
    num_patterns = 2
    sync = True
    flips = 0
    time = 300
    rule = 'DescentL2SolverMulti'
    options = {'incremental' : False, 'tol' : 1e-5, 'lmbd' : 0.5, 'alpha' : 1, 'mu' : 10000.01}
    HN = Asymmetric_Hopfield_network(num_neurons=num_neurons)
    patterns = [random_state(p=0.5, n=num_neurons, values=[0, 1]) for i in range(num_patterns)]
    HN.learn_patterns(patterns, rule, options)
    pattern_r = introduce_random_flips(pattern=patterns[0], k=flips, values=[0, 1])
    retrieved_pattern, data = HN.retrieve_pattern(pattern_r, sync, time, record = True)

    print(f'correlation :')
    print(((2*np.array(patterns) - 1) @ ((2*np.array(patterns) - 1).T)/num_neurons))
    # print('\nSimilarity with different patterns (before): ')
    # for i in range(len(patterns)):
    #     print(np.sum(np.abs((pattern_r.flatten() - patterns[i].flatten()))) / (num_neurons))
    #
    #
    # print('\nSimilarity with different patterns (after): ')
    # for i in range(len(patterns)):
    #     print(np.sum(np.abs((retrieved_pattern.flatten() - patterns[i].flatten()))) / (num_neurons))
    #
    fig1 = plt.figure()
    plt.plot(data['hidden_variables'].T)
    plt.title('H V')
    plt.show()

    fig2 = plt.figure()
    plt.plot(data['u'].T)
    plt.title('adaptation')
    plt.show()

    # fig3 = plt.figure()
    # similarities = data['state'].T
    # plt.plot(similarities)
    # plt.title('State')
    # plt.show()

    fig4 = plt.figure()
    similarities = (((np.array(patterns) - 0.5)*2) @ ((data['state'] - 0.5)*2) / num_neurons).T
    plt.plot(similarities)
    plt.title('Similarity')
    plt.show()