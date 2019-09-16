import numpy as np
from cvxopt import matrix, solvers

def solve_qp(P, q, G, h):
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    sol = solvers.qp(P, q, G=G, h=h, A=None, b=None)
    return sol['x']

# def solve_lp(P, q, G, h):
#     solvers.options['show_progress'] = False
#     P = matrix(P)
#     q = matrix(q)
#     G = matrix(G)
#     h = matrix(h)
#     sol = solvers.qp(P, q, G=G, h=h, A=None, b=None)
#     return sol['x']




def hebbian_lr(N, patterns, incremental, weights = None, biases = None):
    if incremental == True:
        raise AttributeError('The is no separate incremental implementation of Hebb\'s learning rule since it is'
                             ' no different from the nonincremental one in terms of the end result')
    else:
        Z = patterns.T.reshape(N, -1)
        Y = Z @ Z.T
        return (1 / N) * Y

def pseudoinverse(N, patterns, incremental, weights = None, biases = None):
    if incremental == True:
        raise AttributeError('The pseudoiverse learning rule can\'t be incremental')
    else:
        Z = patterns.T.reshape(N, -1)
        Y = Z @ np.linalg.pinv(Z)
        return Y

def storkey_2_order(N, patterns, incremental, weights, biases):
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            h = (weights @ pattern) - (np.diag(weights) * pattern) + biases
            H = np.hstack([h.reshape(-1, 1) for j in range(N)]) - weights * pattern
            pattern_matrix = np.hstack([pattern.reshape(-1, 1) for j in range(N)])
            Z = pattern_matrix - H
            weights += (1 / N) * (Z * Z.T)
        return weights
    else:
        raise (NotImplementedError)

def storkey_simplified(N, patterns, incremental, weights, biases):
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            h = (weights @ pattern) - (np.diag(weights) * pattern) + biases
            weights += (1 / N) * ((pattern - h).reshape(-1, 1) @ (pattern - h).reshape(1, -1))
        return weights
    else:
        Z = patterns.T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        Y = (Z - H) @ (Z - H).T
        weights += (1 / N) * Y
        return weights

def storkey_asymmetric(N, patterns, incremental, weights, biases):
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            h = (weights @ pattern) - (np.diag(weights) * pattern) + biases
            H = np.hstack([h.reshape(-1, 1) for j in range(N)]) - weights * pattern
            pattern_matrix = np.hstack([pattern.reshape(-1, 1) for j in range(N)])
            Z = pattern_matrix - H
            weights += (1 / N) * (pattern_matrix * Z.T)
        return weights
    else:
        Z = patterns.T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        weights += (1 / N) * Z @ (Z - H).T
        return weights

def storkey_original(N, patterns, incremental, weights, biases):
    #TODO : incorporate biases
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            A = (1 / N) * ((pattern.reshape(-1, 1) @ pattern.reshape(1, -1)) - np.identity(N))
            weights += A - A @ weights - weights @ A
        return weights
    else:
        raise (NotImplementedError)

def optimisation_quadratic(N, patterns, incremental = None, weights = None, biases = None):
    epsilon = 1.0
    Z = patterns.T.reshape(N, -1)
    p = Z.shape[-1]
    G = np.zeros((N * p, N ** 2))
    h = -epsilon * np.ones((N * p, 1))
    for j in range(p):
        G[j * N : (j + 1) * N, :] = -Z[:, j].reshape(-1, 1) * np.kron(np.eye(N), Z[:, j].reshape(1, -1))
    # formulate an optimisation task
    P = np.eye(N ** 2)
    q = np.zeros(N ** 2)
    new_weights = solve_qp(P, q, G, h)
    new_weights = np.array(new_weights).reshape(N, N)
    return new_weights

def optimisation_sequential_quadratic(N, patterns, incremental = None, weights = None, biases = None):
    Z = patterns.T.reshape(N, -1)
    for i in range(Z.shape[-1]):
        pattern = Z[:, i].reshape(-1, 1)
        if i == 0:
            weights = np.matmul(pattern, pattern.T)
        else:
            G = np.zeros((N * i, N ** 2))
            h = np.ones((N * i, 1))
            for j in range(i):  # j control the numbers of a pattern
                G[(i - 1) * N : i * N, :] = -pattern.reshape(-1, 1) * np.kron(np.eye(N), pattern.reshape(1, -1))
                h[(i - 1) * N : i * N, :] = -(weights @ pattern.reshape(-1, 1)) * pattern.reshape(-1, 1)
            # formulate an optimisation task
            P = np.eye(N ** 2)
            q = np.zeros(N ** 2)
            delta_w = solve_qp(P, q, G, h)
            delta_w = np.array(delta_w).reshape(N, N)
            weights += delta_w
    return weights