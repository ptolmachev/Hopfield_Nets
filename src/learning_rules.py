import numpy as np
from cvxopt import matrix, solvers


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

# def trace(A,B):
#     return np.sum(A*B.T)


def hebbian_lr(N, patterns, weights, biases, options):
    sc = options['sc']
    incremental = options['incremental']
    if incremental == True:
        raise AttributeError('The is no separate incremental implementation of Hebb\'s learning rule since it is'
                             ' no different from the nonincremental one in terms of the end result')
    else:
        Z = patterns.T.reshape(N, -1)
        Y = Z @ Z.T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        return (1 / N) * Y

def pseudoinverse(N, patterns, weights, biases, options):
    sc = options['sc']
    incremental = options['incremental']
    if incremental == True:
        raise AttributeError('The pseudoiverse learning rule can\'t be incremental')
    else:
        Z = patterns.T.reshape(N, -1)
        Y = Z @ np.linalg.pinv(Z)
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        return Y

def storkey_2_order(N, patterns, weights, biases, options):
    sc = options['sc']
    incremental = options['incremental']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            h = (weights @ pattern) - (np.diag(weights) * pattern) + biases
            H = np.hstack([h.reshape(-1, 1) for j in range(N)]) - weights * pattern
            pattern_matrix = np.hstack([pattern.reshape(-1, 1) for j in range(N)])
            A = pattern_matrix - H
            Y = (A * A.T)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += (1 / N) * Y
        return weights
    else:
        raise (NotImplementedError)

def storkey_simplified(N, patterns, weights, biases, options):
    sc = options['sc']
    incremental = options['incremental']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            h = (weights @ pattern) - (np.diag(weights) * pattern) + biases
            Y = ((pattern - h).reshape(-1, 1) @ (pattern - h).reshape(1, -1))
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += (1 / N) * Y
        return weights
    else:
        Z = patterns.T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        Y = (Z - H) @ (Z - H).T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights += (1 / N) * Y
        return weights

def storkey_asymmetric(N, patterns, weights, biases, options):
    sc = options['sc']
    incremental = options['incremental']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            h = (weights @ pattern) - (np.diag(weights) * pattern) + biases
            H = np.hstack([h.reshape(-1, 1) for j in range(N)]) - weights * pattern
            pattern_matrix = np.hstack([pattern.reshape(-1, 1) for j in range(N)])
            Z = pattern_matrix - H
            Y = (pattern_matrix * Z.T)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += (1 / N) * Y
        return weights
    else:
        Z = patterns.T.reshape(N, -1)
        H = (weights @ Z) + np.hstack([biases.reshape(-1, 1) for i in range(Z.shape[-1])])
        Y = Z @ (Z - H).T
        if sc == False:
            Y[np.arange(N), np.arange(N)] = np.zeros(N)
        weights += (1 / N) * Y
        return weights

def storkey_original(N, patterns, weights, biases, options):
    sc = options['sc']
    incremental = options['incremental']
    Z = patterns.T.reshape(N, -1)
    #TODO : incorporate biases
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i]
            A = (1 / N) * ((pattern.reshape(-1, 1) @ pattern.reshape(1, -1)) - np.identity(N))
            Y = A - A @ weights - weights @ A
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += (1 / N) * Y
        return weights
    else:
        raise (NotImplementedError)

def optimisation_quadratic(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = patterns.T.reshape(N, -1)
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
    return new_weights

def optimisation_linear(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = patterns.T.reshape(N, -1)
    epsilon = 1.0
    Z = patterns.T.reshape(N, -1)
    p = Z.shape[-1]
    G = np.zeros((N * p, N ** 2))
    h = -epsilon * np.ones((N * p, 1))
    #TODO: implement sc deprecation
    for j in range(p):
        G[j * N : (j + 1) * N, :] = -Z[:, j].reshape(-1, 1) * np.kron(np.eye(N), Z[:, j].reshape(1, -1))
    # formulate an optimisation task
    new_weights = l1_minimisation(N, G, h)
    new_weights = np.array(new_weights).reshape(N, N)
    return new_weights

def optimisation_sequential_quadratic(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = patterns.T.reshape(N, -1)
    Z = patterns.T.reshape(N, -1)
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
    return weights

def optimisation_incremental_quadratic(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = patterns.T.reshape(N, -1)
    Z = patterns.T.reshape(N, -1)
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
    return weights

def optimisation_incremental_linear(N, patterns, weights, biases, options):
    sc = options['sc']
    Z = patterns.T.reshape(N, -1)
    epsilon = 1
    for i in range(Z.shape[-1]):
        pattern = Z[:, i].reshape(-1, 1)
        # TODO: implement sc deprecation
        if i == 0:
            weights = np.matmul(pattern, pattern.T)
        else:
            G = -pattern.reshape(-1, 1) * np.kron(np.eye(N), pattern.reshape(1, -1))
            h = -epsilon * np.ones((N, 1)) - (weights @ pattern.reshape(-1, 1)) * pattern.reshape(-1, 1)
            delta_w = l1_minimisation(N, G, h)
            delta_w = np.array(delta_w).reshape(N, N)
            weights += delta_w
    return weights

def descent_l2_norm(N, patterns, weights, biases, options):
    lmbd = 0.5
    incremental = options['incremental']
    sc = options['sc']
    num_it = options['num_it']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i].reshape(-1, 1)
            for j in range(num_it):
                lf = weights @ pattern + biases.reshape(-1,1)
                A = 2 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * (np.tanh(lmbd * lf) - pattern)
                Y = - A @ pattern.T
                delta_b = - A
                if sc == False:
                    Y[np.arange(N), np.arange(N)] = np.zeros(N)
                weights += Y
                biases += delta_b.flatten()
        return weights, biases
    else:
        Z = patterns.T.reshape(N, -1)
        p = Z.shape[-1]
        for j in range(num_it):
            lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
            A = 2 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * (np.tanh(lmbd * lf) - Z)
            Y = - (A @ Z.T)
            delta_b = - np.mean(A, axis = 1)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
            biases += delta_b.flatten()
        return weights, biases

def descent_l2_symm(N, patterns, weights, biases, **options):
    lmbd = 0.5
    incremental = options['incremental']
    sc = options['sc']
    num_it = options['num_it']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i].reshape(-1, 1)
            for j in range(num_it):
                lf = weights @ pattern + biases.reshape(-1,1)
                A = 2 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * (np.tanh(lmbd * lf) - pattern)
                Y = - (A @ pattern.T) - (pattern @ A.T)
                delta_b = - A
                if sc == False:
                    Y[np.arange(N), np.arange(N)] = np.zeros(N)
                weights += Y
                biases += delta_b.flatten()
        return weights, biases
    else:
        Z = patterns.T.reshape(N, -1)
        p = Z.shape[-1]
        for j in range(num_it):
            lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
            A = np.tanh(lmbd * lf) - Z
            Y = - (A @ Z.T) - (Z @ A.T)
            delta_b = - np.mean(A, axis=1)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += 0.1*Y
            biases += 0.1*delta_b.flatten()
        return weights, biases

def descent_overlap(N, patterns, weights, biases, **options):
    lmbd = 0.5
    incremental = options['incremental']
    sc = options['sc']
    num_it = options['num_it']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i].reshape(-1, 1)
            for j in range(num_it):
                lf = weights @ pattern + biases.reshape(-1, 1)
                A = (lmbd * (1 - (np.tanh(lmbd * lf)**2)) * pattern)
                Y = - (A @ pattern.T + pattern @ A.T) / (2 * N)
                if sc == False:
                    Y[np.arange(N), np.arange(N)] = np.zeros(N)
                weights += Y
        return weights
    else:
        Z = patterns.T.reshape(N, -1)
        p = Z.shape[-1]
        for j in range(num_it):
            lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
            A = (lmbd * (1 - (np.tanh(lmbd * lf) ** 2)) * Z)
            Y = - (A @ Z.T + Z @ A.T) / (2 * N)
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
        return weights

def descent_Hamming(N, patterns, weights, biases, **options):
    lmbd = 0.5
    incremental = options['incremental']
    sc = options['sc']
    num_it = options['num_it']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i].reshape(-1, 1)
            for j in range(num_it):
                lf = weights @ pattern + biases.reshape(-1, 1)
                A = (0.5 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * np.sign(np.tanh(lmbd * lf) - pattern))
                Y = - (A @ pattern.T) / N
                if sc == False:
                    Y[np.arange(N), np.arange(N)] = np.zeros(N)
                weights += Y
        return weights
    else:
        Z = patterns.T.reshape(N, -1)
        p = Z.shape[-1]
        for j in range(num_it):
            lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
            A = (0.5 * lmbd * (1 - (np.tanh(lmbd * lf)**2)) * np.sign(np.tanh(lmbd * lf) - Z))
            Y = - (A @ Z.T) / N
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
        return weights

def descent_crossentropy(N, patterns, weights, biases, **options):
    lmbd = 0.5
    incremental = options['incremental']
    sc = options['sc']
    num_it = options['num_it']
    if incremental == True:
        for i in range(patterns.shape[0]):
            pattern = patterns[i].reshape(-1, 1)
            for j in range(num_it):
                lf = weights @ pattern + biases.reshape(-1, 1)
                A = (lmbd * (1 - (np.tanh(lmbd * lf)**2)) * pattern)
                Y = (A @ np.log(2*pattern.T)) / N
                if sc == False:
                    Y[np.arange(N), np.arange(N)] = np.zeros(N)
                weights += Y
        return weights
    else:
        Z = patterns.T.reshape(N, -1)
        p = Z.shape[-1]
        for j in range(num_it):
            lf = weights @ Z + np.hstack([biases.reshape(-1, 1) for i in range(p)])
            A = (lmbd * (1 - (np.tanh(lmbd * lf) ** 2)) * Z)
            Y = ( A @ np.log(2*Z.T) )/ N
            if sc == False:
                Y[np.arange(N), np.arange(N)] = np.zeros(N)
            weights += Y
        return weights
