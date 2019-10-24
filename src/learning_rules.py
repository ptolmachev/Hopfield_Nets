import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from utils import normalise_weights
from scipy.sparse.linalg import lsqr
from math_utils import *
#utils


def hebbian_lr(N, patterns, weights, biases, sc, incremental):
    '''
    Standard Hebbian learning rule
    '''
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
    '''
    the rule proposed in Personnaz et al (1986)
    W = Z Z^+
    '''
    Z = deepcopy(patterns).T.reshape(N, -1)
    Y = N * Z @ np.linalg.pinv(Z)
    weights = Y
    return weights, biases

def l2_difference_minimisation(N, patterns, weights, biases, sc):
    '''
    minimisation of || lmbd*W @ Z - Z||
    '''
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
    '''
    Storkey rule proposed in Storkey,  A.  J.  (1999).
    Efficient  Covariance  Matrix  Methods for  Bayesian  Gaussian  Processes  and  Hop  eld  Neural  Networks.
    '''
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
    '''
    Modification of Storkey rule
    '''
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

def descent_l2_newton(N, patterns, weights, biases, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} (lmbd h_i^k sigma_i^k - sigma_i^k)^2
    '''
    if incremental:
        for i in range(N): #for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                res = minimize(l2norm_difference, x0, args=(pattern, i, lmbd, alpha),
                               jac = l2norm_difference_jacobian, hess=l2norm_difference_hessian,
                               method='Newton-CG', tol=tol, options={'disp' : False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] =  deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N): #for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            res = minimize(l2norm_difference, x0, args=(patterns, i, lmbd, alpha),
                           jac = l2norm_difference_jacobian, hess=l2norm_difference_hessian,
                           method='Newton-CG', tol=tol, options={'disp' : False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] =  deepcopy(res['x'][-1])
    return weights, biases

def descent_l1_newton(N, patterns, weights, biases, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} abs(lmbd h_i^k sigma_i^k - sigma_i^k)
    '''
    if incremental:
        for i in range(N): #for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                res = minimize(l1norm_difference, x0, args=(pattern, i, lmbd, alpha),
                               jac = l1norm_difference_jacobian,
                               method='L-BFGS-B', tol=tol, options={'disp' : False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] =  deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N): #for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            res = minimize(l1norm_difference, x0, args=(patterns, i, lmbd, alpha),
                           jac = l1norm_difference_jacobian,
                           method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] =  deepcopy(res['x'][-1])
    return weights, biases

def descent_crossentropy_newton(N, patterns, weights, biases, incremental, tol, lmbd):
    if incremental:
        for i in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[i].reshape(-1, 1)))
            x0 = np.concatenate([weights.flatten(), biases.flatten()])
            res = minimize(crossentropy, x0, args=(pattern, lmbd), jac = crossentropy_jacobian, method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
            biases = deepcopy(res['x'][N ** 2 :].reshape(-1, 1))
    if incremental == False:
        Z = deepcopy(patterns).T.reshape(N, -1)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(crossentropy, x0, args=(Z, lmbd), jac=crossentropy_jacobian,
                       method='L-BFGS-B', tol=tol, options={'disp': False})
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2:].reshape(-1, 1))
    return weights, biases

def descent_analytical_centre_newton(N, patterns, weights, biases, incremental, tol, lmbd, alpha):
    if incremental:
        for i in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[i].reshape(-1, 1)))
            x0 = np.concatenate([weights.flatten(), biases.flatten()])
            res = minimize(analytical_centre_incremental, x0, args=(pattern, lmbd, alpha, deepcopy(x0)),
                           jac=analytical_centre_incremental_jacobian,
                           method='L-BFGS-B', tol=tol, options={'disp': False})
            weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
            biases = deepcopy(res['x'][N ** 2:].reshape(-1, 1))
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        x0 = np.concatenate([weights.flatten(), biases.flatten()])
        res = minimize(analytical_centre, x0, args=(Z, lmbd, alpha, deepcopy(x0)), jac=analytical_centre_jacobian,
                       method='L-BFGS-B', tol=tol, options={'disp': False})
        weights = deepcopy(res['x'][: N ** 2].reshape(N, N))
        biases = deepcopy(res['x'][N ** 2:].reshape(-1, 1))
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

# def descent_l2_norm(N, patterns, weights, biases, sc, incremental, symm, lmbd, num_it):
#     if incremental == True:
#         for i in range(patterns.shape[0]):
#             # print(f'Learning pattern number {i}')
#             pattern = deepcopy(patterns[i].reshape(-1, 1))
#             A = pattern
#             # while np.linalg.norm(A, 2) >= tol:
#             for j in range(num_it):
#                 lf = weights @ pattern + biases.reshape(-1,1)
#                 A =  2 * lmbd * (lmbd * lf - pattern)
#                 if symm == False:
#                     Y = - A @ pattern.T
#                 else:
#                     Y = - (A @ pattern.T + pattern @ A.T)/2
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
#             A = 2 * lmbd * (1 - (lmbd * lf)**2) * (lmbd * lf - Z)
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

