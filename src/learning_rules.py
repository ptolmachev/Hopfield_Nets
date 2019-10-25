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

def descent_crossentropy_newton(N, patterns, weights, biases, incremental, tol, lmbd, alpha):
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
                res = minimize(crossentropy, x0, args=(pattern, i, lmbd, alpha),
                               jac = crossentropy_jacobian, hess = crossentropy_hessian,
                               method='L-BFGS-B', tol=tol, options={'disp' : False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] =  deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N): #for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            res = minimize(crossentropy, x0, args=(patterns, i, lmbd, alpha),
                           jac = crossentropy_jacobian, hess = crossentropy_hessian,
                           method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] =  deepcopy(res['x'][-1])
    return weights, biases


def descent_analytical_centre_newton(N, patterns, weights, biases, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} exp{-lambda h_i^k sigma_i^k}
    '''
    if incremental:
        for i in range(N):  # for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                res = minimize(analytical_centre, x0, args=(pattern, i, lmbd, alpha),
                               jac=analytical_centre_jacobian, hess=analytical_centre_hessian,
                               method='Newton-CG', tol=tol, options={'disp': False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] = deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N):  # for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            res = minimize(analytical_centre, x0, args=(patterns, i, lmbd, alpha),
                           jac=analytical_centre_jacobian, hess=analytical_centre_hessian,
                           method='Newton-CG', tol=tol, options={'disp': False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] = deepcopy(res['x'][-1])
    return weights, biases

def DO_I(N, patterns, weights, biases, lmbd):
    '''
    rule I described in Diederich and Opper in (1987) Learning of Correlated Patterns in Spin-Glass Networks by Local Learning Rules
    '''
    for i in range(N):  # for each neuron independently
        for j in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[j].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ pattern.T + biases[i])
            # if the  new pattern is not already stable with margin 1
            while (h_i * pattern[i] < 1):
                weights[i, :] = deepcopy(weights[i, :] + lmbd * pattern[i] * pattern)
                biases[i] = deepcopy(biases[i] + lmbd * pattern[i])
                h_i = (weights[i, :] @ pattern.T + biases[i])
    return weights, biases

def DO_II(N, patterns, weights, biases, lmbd, tol):
    '''
    rule II described in Diederich and Opper in (1987) Learning of Correlated Patterns in Spin-Glass Networks by Local Learning Rules
    '''
    for i in range(N):  # for each neuron independently
        for j in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[j].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ pattern.T + biases[i])
            # if the  new pattern is not already stable with margin 1
            while (1 - h_i * pattern[i]) > tol:
                weights[i, :] = deepcopy(weights[i, :] + lmbd * pattern[i] * pattern * (1 - h_i * pattern[i]))
                biases[i] = deepcopy(biases[i] + lmbd * pattern[i])* (1 - h_i * pattern[i])
                h_i = (weights[i, :] @ pattern.T + biases[i])
    return weights, biases


def Krauth_Mezard(N, patterns, weights, biases, lmbd, max_iter):
    '''
    Krauth-Mezard rule proposed in (1987) Krauth Learning algorithms with optimal stability in neural networks
    '''
    Z = np.array(patterns).T
    M = 0
    lf_adjustment_global = (weights @ Z + np.hstack([biases.reshape(N, 1)] * Z.shape[-1])) * Z
    while (np.any(lf_adjustment_global < 0) and M < max_iter):
        for i in range(N):  # for each neuron independently
            # compute overlap (h, sigma)
            lf_adjustment = deepcopy((weights[i, :] @ Z + biases[i]) * Z[i, :])
            #pick the pattern with the weakest overlap
            ind_min = np.argmin(lf_adjustment)
            weakest_pattern = np.array(deepcopy(patterns[ind_min].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ weakest_pattern.T + biases[i])
            # if the  new pattern is not already stable with margin 1
            while (h_i * weakest_pattern[i] < 1):
                weights[i, :] = deepcopy(weights[i, :] + lmbd * weakest_pattern[i] * weakest_pattern)
                biases[i] = deepcopy(biases[i] + lmbd * weakest_pattern[i])
                h_i = (weights[i, :] @ weakest_pattern.T + biases[i])
        lf_adjustment_global = (weights @ Z + np.hstack([biases.reshape(N, 1)] * Z.shape[-1])) * Z
        M += 1
        if M >= max_iter:
            print('Maximum number of iterations has been exceeded')
    return weights, biases

# def Gardner(N, patterns, weights, biases, lmbd, k):
#     '''
#     Gardner rule rule proposed in (1987) Krauth Learning algorithms with optimal stability in neural networks
#     '''
#     for i in range(N):  # for each neuron independently
#         for j in range(patterns.shape[0]):
#             pattern = np.array(deepcopy(patterns[j].reshape(1, N))).squeeze()
#             h_i = (weights[i, :] @ pattern.T + biases[i])
#             y = h_i * pattern[i]/(np.sqrt(np.sum(weights[i,:]**2+ biases[i]**2)))
#             while (y <= k):
#                 weights[i, :] = deepcopy(weights[i, :] + lmbd * pattern[i] * pattern)
#                 biases[i] = deepcopy(biases[i] + lmbd * pattern[i])
#                 h_i = (weights[i, :] @ pattern.T + biases[i])
#                 y = h_i * pattern[i] / (np.sqrt(np.sum(weights[i, :] ** 2 + biases[i] ** 2)))
#     return weights, biases

def Gardner(N, patterns, weights, biases, lmbd, k, max_iter):
    '''
    Gardner rule rule proposed in (1987) Krauth Learning algorithms with optimal stability in neural networks
    '''
    Z = np.array(patterns).T
    M = 0
    y_global = ((weights @ Z + np.hstack([biases.reshape(N, 1)] * Z.shape[-1])).T /(np.sqrt(np.sum(weights ** 2, axis = 1)))).T * Z
    while (np.any(y_global < k) and M < max_iter):
        for i in range(N):  # for each neuron independently
            # compute overlap (h, sigma)
            lf_adjustment = deepcopy(((weights[i, :] @ Z + biases[i])/(np.sqrt(np.sum(weights[i, :] ** 2)))) * Z[i, :])
            #pick the pattern with the weakest overlap
            ind_min = np.argmin(lf_adjustment)
            weakest_pattern = np.array(deepcopy(patterns[ind_min].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ weakest_pattern.T + biases[i])
            # if the  new pattern is not already stable with margin 1
            while (h_i * weakest_pattern[i]/((np.sqrt(np.sum(weights[i, :] ** 2))))  < k):
                weights[i, :] = deepcopy(weights[i, :] + lmbd * weakest_pattern[i] * weakest_pattern)
                biases[i] = deepcopy(biases[i] + lmbd * weakest_pattern[i])
                h_i = (weights[i, :] @ weakest_pattern.T + biases[i])
        y_global = ((weights @ Z + np.hstack([biases.reshape(N, 1)] * Z.shape[-1])).T /(np.sqrt(np.sum(weights ** 2, axis = 1)))).T * Z
        M += 1
        if M >= max_iter:
            print('Maximum number of iterations has been exceeded')
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

