from cvxopt import matrix, solvers
import numpy as np
from copy import deepcopy

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def step(x):
    return np.round(sigmoid(x),0)

def identity_function(x):
    return x

def random_state(p, n, values):
    state = []
    for i in range(n):
        state.append(values[0] if np.random.rand() < p else values[1])
    return np.array(state)

def introduce_random_flips(pattern, k, values):
    new_pattern = deepcopy(pattern)
    inds = np.random.choice(np.arange(len(new_pattern)), k, replace=False)
    for i in range(k):
        new_pattern[inds[i]] = values[0] if new_pattern[inds[i]] == values[1] else values[1]
    return new_pattern

def solve_qp(P, q, G, h):
    solvers.options['show_progress'] = False
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    sol = solvers.qp(P, q, G=G, h=h, A=None, b=None)
    return np.array(sol['x'])

def solve_lp(c, A, b):
    solvers.options['show_progress'] = False
    c = matrix(c)
    A = matrix(A)
    b = matrix(b.flatten())
    sol = solvers.lp(c, A, b)
    return np.array(sol['x'])

def chebyshev_centre(A, b, gamma):
    rows, cols = A.shape
    c = np.zeros(cols + 1)
    c[-1] = -1
    A_ = np.hstack([A, np.sqrt(np.sum(np.power(A, 2), axis=1)).reshape(-1, 1)])
    A_ = np.vstack([A_, -c.reshape(1, -1)])
    b_ = np.append(b, 100).reshape(-1, 1)

    # l2 norm minimisation of w
    P = gamma * np.eye(cols + 1)
    P[:, -1] = P[-1, :] = 0

    res = solve_qp(P=P, q=c, G=A_, h=b_)
    x_c = np.array(res[:-1])
    R = np.float(res[-1])
    return x_c, R

def l1_minimisation(G, h): # l1 norm minimization of x, given inequality constraint Gx <= h
    N = G.shape[-1]
    solvers.options['show_progress'] = False
    c = matrix(np.hstack([np.zeros(N), np.ones(N)]))
    #original constraint
    tmp_1 = np.hstack([G, np.zeros(G.shape)])
    #constraint on absolute value
    tmp_2 = np.kron(np.array([[1, -1],[-1, -1]]), np.eye(N))
    #ensure positivity of t
    tmp_3 = np.hstack([np.zeros(N), -np.ones(N)])
    b = matrix(np.vstack([h.reshape(-1, 1), np.zeros((2 * N, 1)), 0]))
    A = matrix(np.vstack([tmp_1, tmp_2, tmp_3]))
    sol = solvers.lp(c, A, b)
    return sol['x'][:N]

def l2norm_difference(weights_and_biases, patterns, lmbd):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    return np.sum((lmbd * h - Z)**2)

def l2norm_difference_jacobian(weights_and_biases, patterns, lmbd):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    grad_w = ((lmbd * h - Z) * lmbd) @ Z.T + Z @ ((lmbd * h - Z) * lmbd).T
    grad_b = np.sum(lmbd *(lmbd * h - Z), axis=-1)
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])

def l1norm_difference(weights_and_biases, patterns, lmbd):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    return np.sum(np.abs(lmbd * h - Z))

def l1norm_difference_jacobian(weights_and_biases, patterns, lmbd):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    grad_w = 0.5 * ( (np.sign(lmbd * h - Z) * lmbd) @ Z.T + Z @ (np.sign(lmbd * h - Z) * lmbd).T )
    grad_b = np.sum(lmbd * np.sign(lmbd * h - Z), axis=-1)
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])

def crossentropy(weights_and_biases, patterns, lmbd):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    return np.sum( - ((1 + Z)/ 2) * np.log((1 + lmbd * h) / 2) - ((1 - Z) / 2) * np.log((1 - lmbd * h) / 2) )

def crossentropy_jacobian(weights_and_biases, patterns, lmbd):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    grad_w = lmbd * ((lmbd * h - Z)/(1 - lmbd ** 2 * h ** 2)) @ Z.T
    grad_b = np.sum(lmbd * ((lmbd * h - Z)/(1 - lmbd ** 2 * h ** 2)), axis=-1)
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])

def analytical_centre(weights_and_biases, patterns, lmbd, alpha):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    return np.sum(np.exp(-lmbd * Z * h)) + alpha*(np.sum(weights ** 2) + np.sum(biases ** 2))

def analytical_centre_jacobian(weights_and_biases, patterns, lmbd, alpha):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    grad_w = -(lmbd * Z * np.exp(-lmbd * Z * h)) @ Z.T + 2 * alpha * weights
    grad_b = -np.sum((lmbd * Z * np.exp(-lmbd * Z * h)), axis=-1) + 2 * alpha * biases
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])

def analytical_centre_incremental(weights_and_biases, patterns, lmbd, alpha, weights_and_biases_initial):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    weights_initial = weights_and_biases_initial[:N ** 2].reshape(N, N)
    biases_initial = weights_and_biases_initial[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    return np.sum(np.exp(-lmbd * Z * h)) + alpha*(np.sum( (weights - weights_initial) ** 2) + np.sum( (biases - biases_initial) ** 2))

def analytical_centre_incremental_jacobian(weights_and_biases, patterns, lmbd, alpha, weights_and_biases_initial):
    Z = np.array(patterns)
    N = patterns.shape[0]
    p = Z.shape[-1]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    weights_initial = weights_and_biases_initial[:N ** 2].reshape(N, N)
    biases_initial = weights_and_biases_initial[N ** 2:]
    h = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
    grad_w = -(lmbd * Z * np.exp(-lmbd * Z * h)) @ Z.T + 2 * alpha * (weights - weights_initial)
    grad_b = -np.sum((lmbd * Z * np.exp(-lmbd * Z * h)), axis=-1) + 2 * alpha * (biases - biases_initial)
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])