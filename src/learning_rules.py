import numpy as np
from cvxopt import matrix, solvers
from copy import deepcopy
from scipy.optimize import minimize
from utils import normalise_weights
#utils
def solve_qp(P, q, G, h):
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    sol = solvers.qp(P, q, G=G, h=h, A=None, b=None)
    return sol['x']

def l1_minimisation(N, G, h): # l1 norm minimization of x, given inequality constraint Gx <= h
    solvers.options['show_progress'] = False
    c = matrix(np.hstack([np.zeros(N ** 2), np.ones(N ** 2)]))
    tmp_1 = np.kron(np.array([[1, -1],[-1, -1]]), np.eye(N ** 2))
    tmp_2 = np.kron(np.array([[1, 0],[0, 0]]), G)
    tmp_3 = np.hstack([np.zeros(N ** 2), -np.ones(N ** 2)])
    b = matrix(np.vstack([np.zeros((tmp_1.shape[0], 1)), h, np.zeros((h.shape[0], 1)), 0]))
    A = matrix(np.vstack([tmp_1, tmp_2, tmp_3]))
    sol = solvers.lp(c, A, b)
    return sol['x'][:N ** 2]

def identity_function(x):
    return x

def l2norm(weights_and_biases, pattern, activation_function, lmbd):
    N = pattern.shape[0]
    if activation_function == 'identity':
        f = identity_function
    elif activation_function == 'tanh':
        f = np.tanh
    else:
        raise NotImplementedError('You can only use \'identity\' and \'tanh\' activation functions.')

    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    return np.sum((f(lmbd * (weights @ pattern + biases.reshape(-1, 1))) - pattern)**2)

def l2norm_jacobian(weights_and_biases, pattern, activation_function, lmbd):
    N = pattern.shape[0]
    if activation_function == 'identity':
        f = identity_function
        def der_f(x):
            return 1
    elif activation_function == 'tanh':
        f = np.tanh
        def der_f(x):
            return 1 - np.tanh(x) ** 2
    else:
        raise NotImplementedError('You can only use \'identity\' and \'tanh\' activation functions.')

    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ biases.reshape(-1, 1)
    grad_w = 2 * lmbd * ((f(lmbd * h) - pattern) * der_f(lmbd * h)) @ pattern.T
    grad_b = 2 * lmbd *((f(lmbd * h) - pattern) * der_f(lmbd * h))
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])


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

def pseudoinverse(N, patterns, weights, biases, sc):
    Z = deepcopy(patterns).T.reshape(N, -1)
    Y = N * Z @ np.linalg.pinv(Z)
    if sc == False:
        Y[np.arange(N), np.arange(N)] = np.zeros(N)
    weights = Y
    return weights, biases

def storkey_2_order(N, patterns, weights, biases, sc):
    for i in range(patterns.shape[0]):
        weights = normalise_weights(weights)
        pattern = deepcopy(patterns[i]).reshape(-1, 1)
        h = (weights @ pattern + biases.reshape(-1, 1)) - (np.diag(weights).reshape(-1,1) * pattern)
        H = (np.hstack([(h - weights[:, j].reshape(-1, 1) * pattern[j]) for j in range(N)]))
        pattern_matrix = np.hstack([pattern for j in range(N)])
        A = pattern_matrix - H
        Y = (A * A.T)
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights += Y
    return weights, biases

def storkey_simplified(N, patterns, weights, biases, sc, incremental):
    if incremental == True:
        # TODO: the weights blow up!
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            h = deepcopy((weights @ pattern + biases.reshape(-1, 1)) - (np.diag(weights).reshape(-1,1) * pattern))
            #normalizing h
            h = h / np.max(np.abs(h))
            Y = (pattern - h) @ (pattern - h).T
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y

    else:
        weights = normalise_weights(weights)
        Z = deepcopy(patterns).T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        Y = (Z - H) @ (Z - H).T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights +=  Y
    return weights, biases

def storkey_asymmetric(N, patterns, weights, biases, sc, incremental):
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            h = (weights @ pattern + biases.reshape(-1, 1)) - (np.diag(weights).reshape(-1, 1) * pattern)
            # H = (np.hstack([(h - weights[:, j].reshape(-1, 1) * pattern[j]) for j in range(N)]))
            # pattern_matrix = np.hstack([pattern for j in range(N)])
            # Z = pattern_matrix - H
            # Y = (pattern_matrix * Z)
            h = h / np.max(np.abs(h))
            Y = (pattern - h) @ pattern.T + pattern @ (pattern - h).T
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        Y = Z @ (Z - H).T + (Z - H) @ Z .T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights += Y
    return weights, biases

def storkey_original(N, patterns, weights, biases, sc, incremental):
    #TODO: doesnt work?
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            A = (1 / N) * ((pattern @ pattern.T) - np.identity(N))
            Y = A - A @ weights - weights @ A
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
        return weights, biases
    else:
        raise (NotImplementedError)

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
                A =  2 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * (np.tanh(lmbd * lf) - pattern)
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

def descent_l2_with_solver(N, patterns, weights, biases, activation_function, tol, lmbd):
    for i in range(patterns.shape[0]):
        pattern = deepcopy(patterns[i].reshape(-1, 1))
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(l2norm, x0, args=(pattern, activation_function, lmbd), jac = l2norm_jacobian, method='L-BFGS-B', tol=tol)
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
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
