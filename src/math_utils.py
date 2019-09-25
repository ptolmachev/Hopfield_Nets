from cvxopt import matrix, solvers
import numpy as np

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
    if activation_function == 'linear':
        f = identity_function
    elif activation_function == 'tanh':
        f = np.tanh
    else:
        raise NotImplementedError('You can only use \'linear\' and \'tanh\' activation functions.')
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    return np.sum((f(lmbd * (weights @ pattern + biases.reshape(-1, 1))) - pattern)**2)

def l2norm_jacobian(weights_and_biases, pattern, activation_function, lmbd):
    N = pattern.shape[0]
    if activation_function == 'linear':
        f = identity_function
        def der_f(x):
            return 1
    elif activation_function == 'tanh':
        f = np.tanh
        def der_f(x):
            return 1 - np.tanh(x) ** 2
    else:
        raise NotImplementedError('You can only use \'linear\' and \'tanh\' activation functions.')

    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ pattern + biases.reshape(-1, 1)

    # grad_w = 2 * lmbd * ((f(lmbd * h) - pattern) * der_f(lmbd * h)) @ pattern.T
    grad_w = lmbd * ((f(lmbd * h) - pattern) * der_f(lmbd * h)) @ pattern.T \
             + lmbd * pattern @ ((f(lmbd * h) - pattern) * der_f(lmbd * h)).T

    grad_b = lmbd *((f(lmbd * h) - pattern) * der_f(lmbd * h))

    return np.concatenate([grad_w.flatten(), grad_b.flatten()])

def l1norm(weights_and_biases, pattern, activation_function, lmbd):
    N = pattern.shape[0]
    if activation_function == 'linear':
        f = identity_function
    elif activation_function == 'tanh':
        f = np.tanh
    else:
        raise NotImplementedError('You can only use \'linear\' and \'tanh\' activation functions.')
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    return np.sum(np.abs(f(lmbd * (weights @ pattern + biases.reshape(-1, 1))) - pattern))

def l1norm_jacobian(weights_and_biases, pattern, activation_function, lmbd):
    N = pattern.shape[0]
    if activation_function == 'linear':
        f = identity_function
        def der_f(x):
            return 1
    elif activation_function == 'tanh':
        f = np.tanh
        def der_f(x):
            return 1 - np.tanh(x) ** 2
    else:
        raise NotImplementedError('You can only use \'linear\' and \'tanh\' activation functions.')

    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ pattern + biases.reshape(-1, 1)
    # grad_w = lmbd * (np.sign(f(lmbd * h) - pattern) * der_f(lmbd * h)) @ pattern.T
    grad_w = (0.5) * lmbd * ( (np.sign(f(lmbd * h) - pattern) * der_f(lmbd * h)) @ pattern.T \
             +  pattern @ (np.sign(f(lmbd * h) - pattern) * der_f(lmbd * h)).T)
    grad_b = lmbd *(np.sign(f(lmbd * h) - pattern) * der_f(lmbd * h))
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])

def overlap(weights_and_biases, pattern, activation_function, lmbd):
    N = pattern.shape[0]
    if activation_function == 'linear':
        f = identity_function
    elif activation_function == 'tanh':
        f = np.tanh
    else:
        raise NotImplementedError('You can only use \'linear\' and \'tanh\' activation functions.')

    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    return np.sum((f(lmbd * (weights @ pattern + biases.reshape(-1, 1))) * pattern))

def overlap_jacobian(weights_and_biases, pattern, activation_function, lmbd):
    N = pattern.shape[0]
    weights = weights_and_biases[:N ** 2].reshape(N, N)
    biases = weights_and_biases[N ** 2:]
    h = weights @ pattern + biases.reshape(-1, 1)

    if activation_function == 'linear':
        f = identity_function
        grad_w = lmbd * pattern @ pattern.T
        grad_b = lmbd * pattern

    elif activation_function == 'tanh':
        f = np.tanh
        def der_f(x):
            return 1 - np.tanh(x) ** 2
        grad_w = 0.5 * lmbd * (der_f(lmbd * h) * pattern @ pattern.T + pattern @ (der_f(lmbd * h) * pattern).T)
        grad_b = lmbd * der_f(lmbd * h) * pattern
    else:
        raise NotImplementedError('You can only use \'linear\' and \'tanh\' activation functions.')
    return np.concatenate([grad_w.flatten(), grad_b.flatten()])

