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

### L2 NORM ###

def l2norm_difference(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    return (1 / 2) * np.sum((lmbd * h - Z[i, :])**2) + (alpha / 2) * np.sum(weights_and_bias ** 2)

def l2norm_difference_jacobian(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    grad_w_b = ((lmbd * h - Z[i, :]) * lmbd) @ Z_.T + alpha * weights_and_bias
    return grad_w_b

def l2norm_difference_hessian(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    N = Z.shape[0]
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    H = alpha * np.eye(N + 1) + lmbd**2 * Z_ @ Z_.T
    return H

### L1 NORM ###

def l1norm_difference(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    return np.sum(np.abs(lmbd * h - Z[i, :])) + (alpha / 2) * np.sum(weights_and_bias ** 2)

def l1norm_difference_jacobian(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    grad_w_b = (np.sign(lmbd * h - Z[i, :]) * lmbd) @ Z_.T + alpha * weights_and_bias
    return grad_w_b

### CROSSENTROPY

def crossentropy(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    return np.sum( - ((1 + Z[i, :])/ 2) * np.log((1 + lmbd * h) / 2) - ((1 - Z[i, :]) / 2) * np.log((1 - lmbd * h) / 2) ) \
           + (alpha / 2) * np.sum(weights_and_bias ** 2)

def crossentropy_jacobian(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    grad_w_b = lmbd * ((lmbd * h - Z[i, :]) / (1 - lmbd ** 2 * h ** 2)) @ Z_.T + alpha * weights_and_bias
    return grad_w_b

# ADD HESSIAN FOR CROSSENTROPY
def crossentropy_hessian(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    N = Z.shape[0]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    grad_w_b = lmbd * ((lmbd * h - Z[i, :]) / (1 - lmbd ** 2 * h ** 2)) @ Z_.T + alpha * weights_and_bias
    H = alpha * np.eye(N + 1) + lmbd * (Z[i, :]/(1 - lmbd ** 2 * h ** 2) @ Z_.T) - lmbd**3 * (((lmbd * h - Z[i, :]) * h/(1 - lmbd ** 2 * h ** 2))**2) @ Z_.T
    return H

### EXPONENTIAL BARRIER ###

def analytical_centre(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    return np.sum(np.exp(-lmbd * Z[i, :] * h))  + (alpha / 2) * np.sum(weights_and_bias ** 2)

def analytical_centre_jacobian(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    grad_w_b = -(lmbd * Z[i, :] * np.exp(-lmbd * Z[i, :] * h)) @ Z_.T + alpha * weights_and_bias
    return grad_w_b

def analytical_centre_hessian(weights_and_bias, patterns, i, lmbd, alpha):
    Z = np.array(patterns).T
    p = Z.shape[-1]
    N = Z.shape[0]
    # we want to treat the biases as if they are weight from the neurons outside of the network in the state +1
    Z_ = np.vstack([Z, np.ones(p)])
    h = (weights_and_bias.reshape(1, -1) @ Z_).squeeze() # vector of length p
    H = alpha * np.eye(N + 1) + (lmbd**2 * np.exp(-lmbd * Z[i, :] * h)) * Z_ @ Z_.T
    return H
