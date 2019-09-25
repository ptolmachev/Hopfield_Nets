import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from utils import normalise_weights
from scipy.sparse.linalg import lsqr
from math_utils import *
#utils


def hebbian_lr(N, patterns, weights, biases, sc, incremental, unlearning = False, HN = None,
               unlearn_rate = None, num_of_retrieval = None, sync = None, time = None):
    weights = deepcopy(np.zeros((N, N)))
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            Y = (pattern @ pattern.T)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y

            if unlearning == True:
                for r in range(num_of_retrieval):
                    retirieved_pattern = HN.retrieve_pattern(pattern, sync, time, record = False)
                    overlap = np.abs(np.dot(pattern.flatten(), retirieved_pattern) / N)
                    if overlap != 1.0:
                        Y = -unlearn_rate * (1 - overlap) * (retirieved_pattern @ retirieved_pattern.T - np.identity(N))
                        weights += Y
                        weights += (pattern @ pattern.T - np.identity(N))
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        Y = Z @ Z.T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights = Y
    return weights, biases

def pseudoinverse(N, patterns, weights, biases, sc):
    Z = deepcopy(patterns).T.reshape(N, -1)
    Y = N * Z @ np.linalg.pinv(Z)
    if sc == False:
        Y[np.arange(N), np.arange(N)] = np.zeros(N)
    weights = Y
    return weights, biases

def pseudoinverse_nondiag(N, patterns, weights, biases):
    Z = deepcopy(patterns).T.reshape(N, -1)
    p = Z.shape[-1]
    B = np.vstack([np.kron(np.eye(N), Z[:, i].reshape(1, -1)) for i in range(p)])
    D = np.zeros((N, N, N))
    for i in range(N):
        D[i, i, i] = 1
    D = D.reshape(N, -1)
    A = np.vstack([B, D])
    b = np.concatenate([Z.T.flatten(), np.zeros(N)])
    w = lsqr(A, b)[0]
    weights = w.reshape(N, N)
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

def descent_l2_with_solver(N, patterns, weights, biases, activation_function, tol, lmbd):
    for i in range(patterns.shape[0]):
        pattern = deepcopy(patterns[i].reshape(-1, 1))
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l2norm, x0, args=(pattern, activation_function, lmbd), jac = l2norm_jacobian, method='L-BFGS-B', tol=tol)
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    return weights, biases

def descent_l1_with_solver(N, patterns, weights, biases, activation_function, tol, lmbd):
    for i in range(patterns.shape[0]):
        pattern = deepcopy(patterns[i].reshape(-1, 1))
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l1norm, x0, args=(pattern, activation_function, lmbd), jac = l1norm_jacobian, method='L-BFGS-B', tol=tol)
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    return weights, biases

def descent_overlap_with_solver(N, patterns, weights, biases, activation_function, tol, lmbd):
    for i in range(patterns.shape[0]):
        pattern = deepcopy(patterns[i].reshape(-1, 1))
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(overlap, x0, args=(pattern, activation_function, lmbd), jac = overlap_jacobian, method='L-BFGS-B', tol=tol)
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    return weights, biases


def optimisation_quadratic(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = deepcopy(patterns).T.reshape(N, -1)
    epsilon = 1.0
    Z = patterns.T.reshape(N, -1)
    p = Z.shape[-1]
    G = np.zeros((N * p, N ** 2))
    h = -epsilon * np.ones((N * p, 1))
    #TODO: implement sc deprecation
    for j in range(p):
        G[j * N : (j + 1) * N, :] = -Z[:, j].reshape(-1, 1) * np.kron(np.eye(N), Z[:, j].reshape(1, -1))
    # formulate an optimisation task
    P = np.eye(N ** 2)
    q = np.zeros(N ** 2)
    new_weights = solve_qp(P, q, G, h)
    new_weights = np.array(new_weights).reshape(N, N)
    return new_weights, biases

def optimisation_linear(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = patterns.T.reshape(N, -1)
    epsilon = 1.0
    Z = deepcopy(patterns).T.reshape(N, -1)
    p = Z.shape[-1]
    G = np.zeros((N * p, N ** 2))
    h = -epsilon * np.ones((N * p, 1))
    #TODO: implement sc deprecation
    for j in range(p):
        G[j * N : (j + 1) * N, :] = -Z[:, j].reshape(-1, 1) * np.kron(np.eye(N), Z[:, j].reshape(1, -1))
    # formulate an optimisation task
    new_weights = l1_minimisation(N, G, h)
    new_weights = np.array(new_weights).reshape(N, N)
    return new_weights, biases

def optimisation_sequential_quadratic(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = deepcopy(patterns).T.reshape(N, -1)
    epsilon = 1
    for i in range(Z.shape[-1]):
        # TODO: implement sc deprecation
        pattern = Z[:, i].reshape(-1, 1)
        if i == 0:
            weights = np.matmul(pattern, pattern.T)
        else:
            G = np.zeros((N * i, N ** 2))
            h = np.ones((N * i, 1))
            for j in range(i):  # j control the numbers of a pattern
                G[(i - 1) * N : i * N, :] = -pattern.reshape(-1, 1) * np.kron(np.eye(N), pattern.reshape(1, -1))
                h[(i - 1) * N : i * N, :] = -epsilon -(weights @ pattern.reshape(-1, 1)) * pattern.reshape(-1, 1)
            # formulate an optimisation task
            P = np.eye(N ** 2)
            q = np.zeros(N ** 2)
            delta_w = solve_qp(P, q, G, h)
            delta_w = np.array(delta_w).reshape(N, N)
            weights += delta_w
    return weights, biases

def optimisation_incremental_quadratic(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = deepcopy(patterns).T.reshape(N, -1)
    epsilon = 1
    for i in range(Z.shape[-1]):
        pattern = Z[:, i].reshape(-1, 1)
        # TODO: implement sc deprecation
        if i == 0:
            weights = np.matmul(pattern, pattern.T)
        else:
            G = -pattern.reshape(-1, 1) * np.kron(np.eye(N), pattern.reshape(1, -1))
            h = -epsilon * np.ones((N, 1)) - (weights @ pattern.reshape(-1, 1)) * pattern.reshape(-1, 1)
            P = np.eye(N ** 2)
            q = np.zeros(N ** 2)
            delta_w = solve_qp(P, q, G, h)
            delta_w = np.array(delta_w).reshape(N, N)
            weights += delta_w
    return weights, biases

def optimisation_incremental_linear(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = deepcopy(patterns).T.reshape(N, -1)
    epsilon = 1
    for i in range(Z.shape[-1]):
        pattern = Z[:, i].reshape(-1, 1)
        # TODO: implement sc deprecation
        if i == 0:
            weights = pattern @ pattern.T
        else:
            G = -pattern.reshape(-1, 1) * np.kron(np.eye(N), pattern.reshape(1, -1))
            h = -epsilon * np.ones((N, 1)) - (weights @ pattern.reshape(-1, 1)) * pattern.reshape(-1, 1)
            delta_w = l1_minimisation(N, G, h)
            delta_w = np.array(delta_w).reshape(N, N)
            weights += delta_w
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


def descent_overlap(N, patterns, weights, biases, sc, incremental, symm, lmbd, num_it):
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            for j in range(num_it):
                lf = weights @ pattern + biases.reshape(-1, 1)
                A = -(lmbd * (1 - (np.tanh(lmbd * lf)**2)) * pattern)
                if symm == False:
                    Y = - A @ pattern.T
                else:
                    Y = - (A @ pattern.T + pattern @ A.T) / 2
                delta_b = - A
                if sc == False:
                    Y[np.arange(N), np.arange(N)] = np.zeros(N)
                weights += Y
                biases += delta_b.flatten()
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        p = Z.shape[-1]
        for j in range(num_it):
            lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
            A = (lmbd * (1 - (np.tanh(lmbd * lf) ** 2)) * Z)
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
#
# def descent_Hamming(N, patterns, weights, biases, options):
#     lmbd = options['lambda']
#     incremental = options['incremental']
#     sc = options['sc']
#     num_it = options['num_it']
#     symm = options['symm']
#     if incremental == True:
#         for i in range(patterns.shape[0]):
#             pattern = patterns[i].reshape(-1, 1)
#             for j in range(num_it):
#                 lf = weights @ pattern + biases.reshape(-1, 1)
#                 A = (0.5 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * np.sign(np.tanh(lmbd * lf) - pattern)) / N
#                 if symm == False:
#                     Y = - A @ pattern.T
#                 else:
#                     Y = - (A @ pattern.T + pattern @ A.T) / 2
#                 delta_b = - A
#                 if sc == False:
#                     Y[np.arange(N), np.arange(N)] = np.zeros(N)
#                 weights += Y
#                 biases += delta_b.flatten()
#     else:
#         Z = patterns.T.reshape(N, -1)
#         p = Z.shape[-1]
#         for j in range(num_it):
#             lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
#             A = (0.5 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * np.sign(np.tanh(lmbd * lf) - Z)) / N
#             if symm == False:
#                 Y = - (A @ Z.T)
#             else:
#                 Y = - (A @ Z.T + Z @ A.T) / 2
#             delta_b = - np.mean(A, axis = 1)
#             if sc == False:
#                 Y[np.arange(N), np.arange(N)] = np.zeros(N)
#             weights += Y
#             biases += delta_b.flatten()
#     return weights, biases
#
# def descent_crossentropy(N, patterns, weights, biases, options):
#     lmbd = options['lambda']
#     incremental = options['incremental']
#     sc = options['sc']
#     num_it = options['num_it']
#     symm = options['symm']
#     if incremental == True:
#         for i in range(patterns.shape[0]):
#             pattern = patterns[i].reshape(-1, 1)
#             for j in range(num_it):
#                 lf = weights @ pattern + biases.reshape(-1, 1)
#                 A = (lmbd * (1 - pattern*np.tanh(lf)))
#                 if symm == False:
#                     Y = - A @ pattern.T
#                 else:
#                     Y = - (A @ pattern.T + pattern @ A.T) / 2
#                 delta_b = - A
#                 if sc == False:
#                     Y[np.arange(N), np.arange(N)] = np.zeros(N)
#                 weights += Y
#                 biases += delta_b.flatten()
#     else:
#         Z = patterns.T.reshape(N, -1)
#         p = Z.shape[-1]
#         for j in range(num_it):
#             lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
#             A = (lmbd * (1 - Z*np.tanh(lf)))
#             if symm == False:
#                 Y = - (A @ Z.T)
#             else:
#                 Y = - (A @ Z.T + Z @ A.T) / 2
#             delta_b = - np.mean(A, axis = 1)
#             if sc == False:
#                 Y[np.arange(N), np.arange(N)] = np.zeros(N)
#             weights += Y
#             biases += delta_b.flatten()
#     return weights, biases
