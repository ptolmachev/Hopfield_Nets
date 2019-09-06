import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def random_state(p, n):
    return np.array([np.random.choice([-1, 1], 1, [1 - p, p])[0] for i in range(n)])

def introduce_random_flips(pattern, k):
    new_pattern = deepcopy(pattern)
    inds = np.random.choice(np.arange(len(new_pattern)), k, replace=False)
    for i in range(k):
        new_pattern[inds[i]] *= -1
    return new_pattern

def sigmoid(x,T):
    return 1.0/(1 + np.exp(-x/T))

class Hopfield_network():
    def __init__(self, num_neurons, p):
        self.num_neurons = num_neurons
        Y = np.random.randn(self.num_neurons,self.num_neurons)
        R = (Y+Y.T)/2
        R[np.arange(num_neurons),np.arange(num_neurons)] = np.zeros(num_neurons)
        self.weights = (1/self.num_neurons)*R
        self.biases = np.zeros(self.num_neurons)
        self.hidden_state = np.random.rand(num_neurons)-0.5
        self.state = np.sign(self.hidden_state)

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = np.matmul(self.weights, self.state) + self.biases
            self.state = np.sign(self.hidden_state)
        elif sync == False:
            for i in range(num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i,:], self.state) + self.biases[i]
                self.state[i] = np.sign(self.hidden_state[i])
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def learn_patterns(self, patterns, rule='Hebb'):
        patterns = np.array(patterns).squeeze()
        if rule == 'Hebb':
            # self.biases = 1.0*np.mean(patterns, axis = 0)
            Y = np.dot(patterns.T, patterns)
            # deprecate self-connectivity
            Y[np.arange(self.num_neurons),np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights = (1/self.num_neurons)*Y

        elif rule =='Storkey':
            # self.biases = 1.0*np.mean(patterns, axis = 0)
            for i in range(patterns.shape[0]):
                pattern = patterns[i]
                h = np.matmul(self.weights, pattern) - (np.diag(self.weights)*pattern) + self.biases
                H = np.hstack([h.reshape(-1,1) for j in range(self.num_neurons)]) - self.weights*pattern
                pattern_matrix = np.hstack([pattern.reshape(-1,1) for j in range(self.num_neurons)])
                Z = pattern_matrix - H
                self.weights += (1/self.num_neurons)*(Z*Z.T)

        elif rule == 'associative':
            # self.biases = 1.0 * np.mean(patterns, axis=0)
            Y = np.matmul(patterns.T, np.linalg.pinv(patterns.T))
            Y[np.arange(self.num_neurons), np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights = (1/self.num_neurons) * Y
        else:
            raise ValueError('the parameter rule can only take values \'Hebb\' or \'Storkey\'')
        return None

    def set_weights(self, new_weights):
        self.weights = new_weights
        return None

    def retrieve_pattern(self, new_state, sync, time, record = False):
        self.state = new_state
        t = 0
        if record == True:
            data = dict()
            data['hidden_variables'] = []
            data['state'] = []
        while t < time:
            self.update_state(sync)
            t += 1
            if record == True:
                data['hidden_variables'].append(deepcopy(self.hidden_state))
                data['state'].append(deepcopy(self.state))
        if record == True:
            data['hidden_variables'] = np.array(data['hidden_variables']).T
            data['state'] = np.array(data['state']).T
            return self.state, data
        else:
            return self.state

class Boltzmann_machine(Hopfield_network):
    def __init__(self, num_neurons, T_max, T_decay, p):
        super().__init__(num_neurons, p)
        self.T = T_max
        self.T_max = T_max
        self.T_decay = T_decay

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = np.matmul(self.weights,self.state) + self.biases
            h_tmp = sigmoid(self.hidden_state, self.T)
            self.state = np.array([(1 if np.random.rand() < h_tmp[i] else -1) for i in range(len(h_tmp))])
        elif sync == False:
            for i in range(num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i,:], self.state)
                self.state[i] = 1 if np.random.rand() < sigmoid(self.hidden_state[i], self.T) else -1
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def retrieve_pattern(self, new_state, sync, time = None, record = False):
        self.T = self.T_max
        self.state = new_state
        if record == True:
            data = dict()
            data['hidden_variables'] = []
            data['state'] = []
        while self.T > 0.0001:
            self.update_state(sync)
            self.T *= self.T_decay
            if record == True:
                data['hidden_variables'].append(deepcopy(self.hidden_state))
                data['state'].append(deepcopy(self.state))
        if record == True:
            data['hidden_variables'] = np.array(data['hidden_variables']).T
            data['state'] = np.array(data['state']).T
            return self.state, data
        else:
            return self.state


class Continuous_HN(Hopfield_network):
    def __init__(self, num_neurons, T,alpha, p):
        super().__init__(num_neurons, p)
        self.T = T
        self.alpha = alpha
        self.fr = (self.state + 1)/2

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = self.hidden_state + self.alpha*(-self.hidden_state + np.matmul(self.weights,self.state) + self.biases)
            self.state = 2*sigmoid(self.hidden_state, self.T) - 1
        else:
            for i in range(self.num_neurons):
                self.hidden_state[i] = self.hidden_state[i] + self.alpha * (-self.hidden_state[i] + np.dot(self.weights[i,:], self.state) + self.biases[i])
                self.state[i] = 2*sigmoid(self.hidden_state[i], self.T) - 1
        self.fr = (self.state + 1) / 2
        return None

class Continuous_HN_adaptive(Hopfield_network):
    def __init__(self, num_neurons, T, alpha, beta, p):
        super().__init__(num_neurons, p)
        self.T = T
        self.alpha = alpha
        self.fr = (self.state + 1)/2
        self.u = np.zeros(self.num_neurons)
        self.beta = beta

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state += self.alpha*(-self.hidden_state + np.matmul(self.weights,self.state) - self.u + self.biases)
            self.u += self.beta*(self.hidden_state-self.u)
            self.state = 2*sigmoid(self.hidden_state, self.T) - 1
        else:
            for i in range(self.num_neurons):
                self.hidden_state[i] += self.alpha * (-self.hidden_state[i] + np.dot(self.weights[i,:], self.state) - self.u[i] + self.biases[i])
                self.u += self.beta * (self.hidden_state[i] - self.u[i])
                self.state[i] = 2*sigmoid(self.hidden_state[i], self.T) - 1
        self.fr = (self.state + 1) / 2
        return None

if __name__ == '__main__':
    num_neurons = 150
    num_patterns = 30
    sync = True
    rule = 'Storkey' #'Storkey'
    flips = 40
    time = 500
    # HN = Hopfield_network(num_neurons=num_neurons, p=0.5)
    # HN = Boltzmann_machine(num_neurons=num_neurons,T_max=0.1, T_decay=0.99, p=0.5)
    # HN = Continuous_HN(num_neurons=num_neurons, T=0.000000001, alpha=0.99, p=0.5)
    HN = Continuous_HN_adaptive(num_neurons=num_neurons, T=0.2, alpha=0.1, beta = 0.00, p=0.5)
    patterns = [random_state(0.5, num_neurons) for i in range(num_patterns)]
    HN.learn_patterns(patterns, rule=rule)

    pattern_r = introduce_random_flips(patterns[0], flips)

    print('\nSimilarity with different patterns (before): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(pattern_r - patterns[i], 2))

    retrieved_pattern, data = HN.retrieve_pattern(pattern_r, sync, time, record = True)
    print('\nSimilarity with different patterns (after): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(retrieved_pattern - patterns[i], 2))

    fig1 = plt.figure()
    plt.plot(data['hidden_variables'].T)
    plt.show()
    fig3 = plt.figure()
    similarities = np.array([[np.dot(data['state'][:,j],patterns[i])/num_neurons for i in range(len(patterns))] for j in range(time)])
    plt.plot(similarities)
    plt.show()
    # print('\nSimilarity of signs with different patterns (after): ')
    # for i in range(len(patterns)):
    #     print(np.linalg.norm(np.sign(retrieved_pattern) - patterns[i], 2))

