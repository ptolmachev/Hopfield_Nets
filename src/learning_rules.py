# import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian
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

def storkey(N, patterns, weights, biases, sc, incremental):
    '''
    Storkey rule proposed in Storkey,  A.  J.  (1999).
    Efficient  Covariance  Matrix  Methods for  Bayesian  Gaussian  Processes  and  Hopfield  Neural  Networks.
    '''
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = deepcopy(patterns[i]).reshape(-1, 1)
            h = weights @ pattern + biases.reshape(-1, 1)
            Y = (1 / N) * ((pattern @ pattern.T - np.identity(N)) - h @ pattern.T - pattern @ h.T + h @ h.T)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
    else:
        Z = deepcopy(patterns).T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        Y = (1 / N) * ((Z @ Z.T - np.identity(N)) - H @ Z.T - Z @ H.T + H @ H.T)
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

def descent_l2(N, patterns, weights, biases, sc, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} (lmbd h_i^k sigma_i^k - sigma_i^k)^2

    works better with L-BFGS-B
    '''
    jac = egrad(l2norm_difference, 0)
    # hess = jacobian(jac)
    if incremental:
        for i in range(N): #for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100*np.ones(x0.shape[-1]), 100*np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(l2norm_difference, x0, args=(pattern, i, lmbd, alpha),
                               jac = jac,# hess=hess,
                               bounds = bnds,
                               method='L-BFGS-B', tol=tol, options={'disp' : False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] =  deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N): #for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-100*np.ones(x0.shape[-1]),100*np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(l2norm_difference, x0, args=(patterns, i, lmbd, alpha),
                           jac = jac, #hess=hess,
                           bounds=bnds,
                           method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] =  deepcopy(res['x'][-1])
    return weights, biases


def descent_l1(N, patterns, weights, biases, sc, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} abs(lmbd h_i^k sigma_i^k - sigma_i^k)

    '''
    jac = egrad(l1norm_difference, 0)
    if incremental:
        for i in range(N): #for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100 * np.ones(x0.shape[-1]), 100 * np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(l1norm_difference, x0, args=(pattern, i, lmbd, alpha),
                               jac = jac,
                               bounds=bnds,
                               method='L-BFGS-B', tol=tol, options={'disp' : False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] =  deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N): #for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-100*np.ones(x0.shape[-1]), 100*np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(l1norm_difference, x0, args=(patterns, i, lmbd, alpha),
                           jac = jac,
                           bounds=bnds,
                           method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] =  deepcopy(res['x'][-1])
    return weights, biases

def descent_crossentropy(N, patterns, weights, biases, sc, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} abs(lmbd h_i^k sigma_i^k - sigma_i^k)
    '''
    jac = egrad(crossentropy, 0)
    # hess = jacobian(jac)
    if incremental:
        for i in range(N): #for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100 * np.ones(x0.shape[-1]), 100 * np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(crossentropy, x0, args=(pattern, i, lmbd, alpha),
                               jac=jac, #hess=hess,
                               bounds=bnds,
                               method='L-BFGS-B', tol=tol, options={'disp' : False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] =  deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N): #for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-100*np.ones(x0.shape[-1]), 100*np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(crossentropy, x0, args=(patterns, i, lmbd, alpha),
                           jac=jac,# hess=hess,
                           bounds=bnds,
                           method='L-BFGS-B', tol=tol, options={'disp' : False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] =  deepcopy(res['x'][-1])
    return weights, biases


def descent_exp_barrier(N, patterns, weights, biases, sc, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} exp{-lambda h_i^k sigma_i^k}

    comment: for some reson L-BFGS-B without a hessian works much faster than Newton-CG with Hessian!
    '''
    jac = egrad(sum_exp_barriers, 0)
    # hess = jacobian(jac)
    if incremental:
        for i in range(N):  # for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100 * np.ones(x0.shape[-1]), 100 * np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(sum_exp_barriers, x0, args=(pattern, i, lmbd, alpha),
                               jac=jac,# hess=hess,
                               bounds=bnds,
                               method='L-BFGS-B', tol=tol, options={'disp': False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] = deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N):  # for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-100 * np.ones(x0.shape[-1]), 100 * np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(sum_exp_barriers, x0, args=(patterns, i, lmbd, alpha),
                           jac=jac,# hess=hess,
                           bounds=bnds,
                           method='L-BFGS-B', tol=tol, options={'disp': False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] = deepcopy(res['x'][-1])
    return weights, biases

def descent_exp_barrier_si(N, patterns, weights, biases, sc, incremental, tol, lmbd):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} -(h_i^{\mu} \sigma_i^{\mu}) / (\sum_j w_{ij}^2 + b_i^2)^(0.5)

    comment: for some reson L-BFGS-B without a hessian works much faster than Newton-CG with Hessian!
    '''
    jac = egrad(sum_exp_barriers_si, 0)
    # hess = jacobian(jac)
    if incremental:
        for i in range(N):  # for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100*np.ones(x0.shape[-1]), 100*np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(sum_exp_barriers_si, x0, args=(pattern, i, lmbd),
                               jac= jac,# hess = hess,
                               bounds = bnds,
                               method='L-BFGS-B', tol=tol, options={'disp': False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] = deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N):  # for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-np.ones(x0.shape[-1]),np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(sum_exp_barriers_si, x0, args=(patterns, i, lmbd),
                           jac= jac, #hess = hess,
                           bounds = bnds,
                           method='L-BFGS-B', tol=tol, options={'disp': False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] = deepcopy(res['x'][-1])
    return weights, biases

def descent_overlap_si(N, patterns, weights, biases, sc, incremental, tol, lmbd, alpha):
    '''
    Newton's method for minimising \sum_{k = 1}^{p} -(h_i^{\mu} \sigma_i^{\mu}) / (\sum_j w_{ij}^2 + b_i^2)^(0.5)
    '''
    jac = egrad(overlap_si, 0)
    hess = jacobian(jac)
    if incremental:
        for i in range(N):  # for each neuron independently
            for j in range(patterns.shape[0]):
                pattern = np.array(deepcopy(patterns[j].reshape(1, N)))
                w_i = weights[i, :]
                b_i = biases[i]
                x0 = np.append(w_i, b_i)
                bnds = list(zip(-100 * np.ones(x0.shape[-1]), 100 * np.ones(x0.shape[-1])))
                if sc == False:
                    bnds[i] = (0, 0)
                res = minimize(overlap_si, x0, args=(pattern, i, lmbd, alpha),
                               jac=jac, hess=hess,
                               bounds=bnds,
                               method='Newton-CG', tol=tol, options={'disp': False})
                weights[i, :] = deepcopy(res['x'][:-1])
                biases[i] = deepcopy(res['x'][-1])
    if incremental == False:
        patterns = np.array(deepcopy(patterns.reshape(-1, N)))
        for i in range(N):  # for each neuron independently
            w_i = weights[i, :]
            b_i = biases[i]
            x0 = np.append(w_i, b_i)
            bnds = list(zip(-100 * np.ones(x0.shape[-1]), 100 * np.ones(x0.shape[-1])))
            if sc == False:
                bnds[i] = (0, 0)
            res = minimize(overlap_si, x0, args=(patterns, i, lmbd, alpha),
                           jac=jac, hess=hess,
                           bounds=bnds,
                           method='Newton-CG', tol=tol, options={'disp': False})
            weights[i, :] = deepcopy(res['x'][:-1])
            biases[i] = deepcopy(res['x'][-1])
    return weights, biases

def DiederichOpper_I(N, patterns, weights, biases, sc, lr):
    '''
    rule I described in Diederich and Opper in (1987) Learning of Correlated Patterns in Spin-Glass Networks by Local Learning Rules
    '''
    for i in range(N):  # for each neuron independently
        for j in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[j].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ pattern.T + biases[i])
            # if the  new pattern is not already stable with margin 1
            while (h_i * pattern[i] < 1):
                weights[i, :] = deepcopy(weights[i, :] + lr * pattern[i] * pattern)
                biases[i] = deepcopy(biases[i] + lr * pattern[i])
                if sc == True:
                    weights[i, i] = 0
                h_i = (weights[i, :] @ pattern.T + biases[i])
    return weights, biases

def DiederichOpper_II(N, patterns, weights, biases, sc, lr, tol):
    '''
    rule II described in Diederich and Opper in (1987) Learning of Correlated Patterns in Spin-Glass Networks by Local Learning Rules
    '''
    for i in range(N):  # for each neuron independently
        for j in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[j].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ pattern.T + biases[i])
            # if the  new pattern is not already stable with margin 1
            while (np.abs(1 - h_i * pattern[i])) > tol:
                weights[i, :] = deepcopy(weights[i, :] + lr * pattern[i] * pattern * (1 - h_i * pattern[i]))
                biases[i] = deepcopy(biases[i] + lr * pattern[i])* (1 - h_i * pattern[i])
                if sc == True:
                    weights[i, i] = 0
                h_i = (weights[i, :] @ pattern.T + biases[i])
    return weights, biases


def Krauth_Mezard(N, patterns, weights, biases, sc, lr, max_iter):
    '''
    Krauth-Mezard rule proposed in (1987) Krauth Learning algorithms with optimal stability in neural networks
    '''
    Z = np.array(patterns).T
    M = 0
    lf_alignment_global = (weights @ Z + np.hstack([biases.reshape(N, 1)] * Z.shape[-1])) * Z
    # lf_alignment_global: if any of these numbers are below zero, some of the patterns is not stable
    while (np.any(lf_alignment_global < 0) and (M <= max_iter)):
        for i in range(N):  # for each neuron independently
            # compute local field alignment (h, sigma)
            lf_alignment = deepcopy((weights[i, :] @ Z + biases[i]) * Z[i, :])
            # pick the pattern with the weakest lf alignment
            ind_min = np.argmin(lf_alignment)
            weakest_pattern = np.array(deepcopy(patterns[ind_min].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ weakest_pattern.T + biases[i])
            # if the  new pattern is not already stable with margin 1
            # it doesnt really matter what exact margin is here, as weights could be scaled by an arbitrary positive factor
            while (h_i * weakest_pattern[i] < 1):
                # add bisection search here?
                weights[i, :] = deepcopy(weights[i, :] + lr * weakest_pattern[i] * weakest_pattern)
                biases[i] = deepcopy(biases[i] + lr * weakest_pattern[i])
                if sc == True:
                    weights[i, i] = 0
                h_i = (weights[i, :] @ weakest_pattern.T + biases[i])
        lf_alignment_global = (weights @ Z + np.hstack([biases.reshape(N, 1)] * Z.shape[-1])) * Z
        M += 1
        if M >= max_iter:
            print('Maximum number of iterations has been exceeded')

    #rescale weights and biases:
    normalising_factor = np.maximum(np.max(np.abs(weights)),np.max(np.abs(biases)))
    weights = weights / normalising_factor
    biases = biases / normalising_factor

    return weights, biases

def Gardner(N, patterns, weights, biases, sc, lr, k):
    '''
    Gardner rule rule proposed in (1988) The space of interactions in neural network models
    '''
    for i in range(N):  # for each neuron independently
        for j in range(patterns.shape[0]):
            pattern = np.array(deepcopy(patterns[j].reshape(1, N))).squeeze()
            h_i = (weights[i, :] @ pattern.T + biases[i])
            sum_of_squares = np.sum(weights[i, :] ** 2) + biases[i] ** 2
            y = (h_i * pattern[i])/(np.sqrt(sum_of_squares))
            while (k >= y):
                weights[i, :] = deepcopy(weights[i, :] + lr * pattern[i] * pattern)
                if sc == True:
                    weights[i, i] = 0
                biases[i] = deepcopy(biases[i] + lr * pattern[i])
                h_i = (weights[i, :] @ pattern.T + biases[i])
                sum_of_squares = np.sum(weights[i,:]**2) + biases[i]**2
                y = (h_i * pattern[i])/(np.sqrt(sum_of_squares))
    return weights, biases

def Gardner_Krauth_Mezard(N, patterns, weights, biases, sc, lr, k, max_iter):
    '''
    Gardner rule rule proposed in (1987) Krauth Learning algorithms with optimal stability in neural networks +
    Krauth Mezard update strategy
    '''
    Z = np.array(patterns).T
    M = 0
    p = Z.shape[-1]
    Z_ = np.vstack([Z, np.ones(p)])
    w_and_b = deepcopy(np.hstack([weights, biases.reshape(N, 1)]))
    y_global = ( (w_and_b @ Z_).T/ (np.sqrt(np.sum(w_and_b ** 2, axis=1))) )* Z.T #
    while (np.any(y_global < k) and M < max_iter):
        for i in range(N):  # for each neuron independently
            # compute normalised stability measure (h_i, sigma_i)/|w_i|^2_2
            sum_of_squares = np.sum(weights[i, :] ** 2 + biases[i]**2)
            ys =  ( (weights[i, :] @ Z + biases[i])/ (np.sqrt(sum_of_squares)) )  * Z[i, :] #
            #pick the pattern with the weakest y
            ind_min = np.argmin(ys)
            weakest_pattern = np.array(deepcopy(patterns[ind_min].reshape(1, N)))
            h_i = (weights[i, :].reshape(1, N) @ weakest_pattern.T + biases[i]).squeeze()
            # if the new weakest pattern is not yet stable with the margin k
            y = (h_i * weakest_pattern[0, i])/(np.sqrt(sum_of_squares)) #
            while (y < k):
                weights[i, :] = deepcopy(weights[i, :] + lr * (weakest_pattern[0, i] * weakest_pattern).squeeze())
                #set diagonal elements to zero
                if sc == True:
                    weights[i, i] = 0
                biases[i] = biases[i] + lr * weakest_pattern[0, i]
                sum_of_squares = np.sum(weights[i, :] ** 2 + biases[i] ** 2)
                h_i = (weights[i, :].reshape(1, N) @ weakest_pattern.T + biases[i]).squeeze()
                y = (h_i * weakest_pattern[0, i])/(np.sqrt(sum_of_squares)) #
        w_and_b = deepcopy(np.hstack([weights, biases.reshape(N, 1)]))
        y_global = ( (w_and_b @ Z_).T/ (np.sqrt(np.sum(w_and_b ** 2, axis=1))) )* Z.T #
        M += 1
        if M >= max_iter:
            print('Maximum number of iterations has been exceeded')
    return weights, biases



