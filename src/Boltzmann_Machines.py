from math_utils import *
from Hopfield_Network import Hopfield_network
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr

################# BOLTZMANN MACHINE #################
class Boltzmann_machine(Hopfield_network):
    def __init__(self, num_neurons, T_max, T_decay):
        super().__init__(num_neurons)
        self.T = T_max
        self.T_max = T_max
        self.T_decay = T_decay

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = (self.weights @ self.state.reshape(-1, 1) + self.biases.reshape(-1, 1)).flatten()
            h_tmp = sigmoid(self.hidden_state/self.T)
            self.state = np.array([(1 if np.random.rand() < h_tmp[i] else -1) for i in range(len(h_tmp))])
        elif sync == False:
            for i in range(num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i, :], self.state)
                self.state[i] = 1 if np.random.rand() < sigmoid(self.hidden_state[i] / self.T) else -1
        else:
            raise AttributeError('\'sync\' variable can take only boolean values')
        return None

    def retrieve_pattern(self, new_state, sync, time = None, record = False):
        self.T = self.T_max
        self.state = new_state
        if record == True:
            data = dict()
            data['hidden_variables'] = []
            data['state'] = []
        while self.T > 0.0001:
            self.update_state(sync)
            self.T *= self.T_decay
            if record == True:
                data['hidden_variables'].append(deepcopy(self.hidden_state))
                data['state'].append(deepcopy(self.state))
        if record == True:
            data['hidden_variables'] = np.array(data['hidden_variables']).T
            data['state'] = np.array(data['state']).T
            return self.state, data
        else:
            return self.state

#LEARNING RULES

def hebbian_lr(N, patterns, weights, biases, sc, incremental):
    weights = deepcopy(np.zeros((N, N)))
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            Y = (pattern @ pattern.T)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        Y = Z @ Z.T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights = Y
    return weights, biases

def pseudoinverse(N, patterns, weights, biases):
    Z = deepcopy(patterns).T.reshape(N, -1)
    Y = N * (Z) @ np.linalg.pinv(Z)
    weights = Y
    return weights, biases

def l2_difference_minimisation(N, patterns, weights, biases, sc):
    Z = deepcopy(patterns).T.reshape(N, -1)
    W = np.array([])
    for i in range(N):
        A = Z.T
        b = Z[i, :]
        if sc == False:
            # add one more constraint to eliminate the diagonals
            tmp = np.zeros((1, N))
            tmp[0, i] = 1
            A = np.vstack([A, tmp])
            b = np.append(b, 0)
        elif sc == True:
            pass
        else:
            raise AttributeError('The \'sc\' parameter can take only boolean values')
        w = lsqr(A, b)[0]
        W = np.append(W, w)
    weights = W.reshape(N, N)
    return weights, biases

def storkey(N, patterns, weights, biases, sc, incremental, order):
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            h = weights @ pattern + biases.reshape(-1, 1)
            if order == 1:
                Y = (1 / N) * ( (pattern @ pattern.T - np.identity(N)) - h @ pattern.T - pattern @ h.T)
            elif order == 2:
                Y = (1 / N) * ((pattern @ pattern.T - np.identity(N)) - h @ pattern.T - pattern @ h.T + h @ h.T)
            else:
                raise AttributeError('Order of Storkey rule could be either \'1\' or \'2\'')
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        if order == 1:
            Y = (1 / N) * ( (Z @ Z.T - np.identity(N)) - H @ Z.T - Z @ H.T)
        elif order == 2:
            Y = (1 / N) * ((Z @ Z.T - np.identity(N)) - H @ Z.T - Z @ H.T + H @ H.T)
        else:
            raise AttributeError('Order of Storkey rule could be either \'1\' or \'2\'')
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights += Y
    return weights, biases

def storkey_normalised_lf(N, patterns, weights, biases, sc, incremental):
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            h = (weights @ pattern + biases.reshape(-1, 1)) - (np.diag(weights).reshape(-1, 1) * pattern)
            h = h / np.max(np.abs(h))
            Y = ((pattern - h) @ pattern.T + pattern @ (pattern - h).T) / 2
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        # scaling
        H = (H.T/np.max(np.abs(H.T), axis = 1, keepdims=True)).T
        Y = ((Z - H) @ Z.T + Z @ (Z - H).T) / 2
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights += Y
    return weights, biases

def descent_l2_with_solver(N, patterns, weights, biases, incremental, tol, lmbd):
    if incremental:
        for i in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[i].reshape(-1, 1)))
            x0 = np.concatenate([weights.flatten(), biases.flatten()])
            res = minimize(l2norm_difference, x0, args=(pattern, lmbd), jac = l2norm_difference_jacobian, method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
            biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    if incremental == False:
        Z = deepcopy(patterns).T.reshape(N, -1)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l2norm_difference, x0, args=(Z, lmbd), jac=l2norm_difference_jacobian,
                       method='L-BFGS-B', tol=tol, options={'disp': False})
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2:].reshape(-1, 1))
    return weights, biases

def descent_l1_with_solver(N, patterns, weights, biases, activation_function, tol, lmbd):
    for i in range(patterns.shape[0]):
        pattern = deepcopy(patterns[i].reshape(-1, 1))
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l1norm_difference, x0, args=(pattern, activation_function, lmbd), jac = l1norm_difference_jacobian, method='L-BFGS-B', tol=tol)
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    return weights, biases

def find_chebyshev_centre(N, patterns, weights, biases, gamma):
    Z = deepcopy(patterns).T.reshape(N, -1)
    p = Z.shape[-1]
    W = np.array([])
    # separate the optimization task into subproblems for each neuron
    for i in range(N):
        A = -(Z * Z[i, :]).T
        b = 0.0 * np.ones(p)
        x_c, R = chebyshev_centre(A, b, gamma)
        W = np.append(W, x_c)
    weights = W.reshape(N, N)
    return weights, biases

def descent_l2_norm(N, patterns, weights, biases, sc, incremental, symm, lmbd, num_it):
    if incremental == True:
        for i in range(patterns.shape[0]):
            # print(f'Learning pattern number {i}')
            pattern = deepcopy(patterns[i].reshape(-1, 1))
            A = pattern
            # while np.linalg.norm(A, 2) >= tol:
            for j in range(num_it):
                lf = weights @ pattern + biases.reshape(-1,1)
                A =  2 * lmbd * (np.tanh(lmbd * lf) - pattern) #(1 - (np.tanh(lmbd * lf)**2))
                if symm == False:
                    Y = - A @ pattern.T
                else:
                    Y = - (A @ pattern.T + pattern @ A.T)/2
                delta_b = - A
                if sc == False:
                    Y[np.arange(N), np.arange(N)] = np.zeros(N)
                weights += Y
                biases += delta_b.flatten()
    else:
        Z = patterns.T.reshape(N, -1)
        p = Z.shape[-1]
        for j in range(num_it):
            lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
            A = 2 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * (np.tanh(lmbd * lf) - Z)
            if symm == False:
                Y = - (A @ Z.T)
            else:
                Y = - (A @ Z.T + Z @ A.T) / 2
            delta_b = - np.mean(A, axis = 1)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
            biases += delta_b.flatten()
    return weights, biases


if __name__ == '__main__':
    num_neurons = 100
    num_patterns = 5
    sync = True
    flips = 20
    time = 100
    rule = 'DescentL1Solver'
    options = {'tol' : 1e-3, 'lmbd' : 0.5}#{'sc' : True, 'incremental' : True}#{}#
    HN = Boltzmann_machine(num_neurons=num_neurons, T_max=10, T_decay=0.9)
    patterns = [random_state(p=0.5, n=num_neurons, values=[-1, 1]) for i in range(num_patterns)]
    HN.learn_patterns(patterns, rule, options)
    pattern_r = introduce_random_flips(pattern=patterns[2], k=flips, values=[-1, 1])

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