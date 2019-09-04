import numpy as np
from copy import deepcopy

def random_state(p, n):
    return np.array([np.random.choice([-1, 1], 1, [1 - p, p])[0] for i in range(n)])

def introduce_random_flips(pattern, k):
    new_pattern = deepcopy(pattern)
    inds = np.random.choice(np.arange(len(new_pattern)), k)
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
        self.hidden_state = np.random.rand(num_neurons)-0.5
        self.state = np.sign(self.hidden_state)

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = (np.dot(self.weights,self.state.reshape(-1,1))).flatten()
            self.state = np.sign(self.hidden_state)
        elif sync == False:
            for i in range(num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i,:], self.state)
                self.state[i] = np.sign(self.hidden_state[i])
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def learn_patterns(self, patterns, rule='Hebb'):
        patterns = np.array(patterns).squeeze()
        if rule == 'Hebb':
            Y = np.dot(patterns.T, patterns)
            # deprecate self-connectivity
            Y[np.arange(self.num_neurons),np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights = (1/self.num_neurons)*np.dot(patterns.T, patterns)
        elif rule =='Storkey':
            # self.weights = np.zeros((self.num_neurons, self.num_neurons))
            for i in range(patterns.shape[0]):
                pattern = patterns[i]
                h = (np.dot(self.weights, pattern.reshape(-1,1)).T).flatten() - (np.diag(self.weights)*pattern)
                H = np.hstack([h.reshape(-1,1) for j in range(self.num_neurons)]) - self.weights*pattern
                pattern_matrix = np.hstack([pattern.reshape(-1,1) for j in range(self.num_neurons)])
                Z = pattern_matrix - H
                self.weights += (1/self.num_neurons)*(Z*Z.T)
        return None

    def set_weights(self, new_weights):
        self.weights = new_weights
        return None

    def retrieve_pattern(self, new_state, sync, time):
        self.state = new_state
        t = 0
        while t < time:
            self.update_state(sync)
            t += 1
        return self.state

class Boltzmann_machine(Hopfield_network):
    def __init__(self, num_neurons, T_max, T_decay, p):
        super().__init__(num_neurons, p)
        self.T = T_max
        self.T_max = T_max
        self.T_decay = T_decay

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = (np.dot(self.weights,self.state.reshape(-1,1))).flatten()
            h_tmp = sigmoid(self.hidden_state, self.T)
            self.state = np.array([(1 if np.random.rand() < h_tmp[i] else -1) for i in range(len(h_tmp))])
        elif sync == False:
            for i in range(num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i,:], self.state)
                self.state[i] = 1 if np.random.rand() < sigmoid(self.hidden_state[i], self.T) else -1
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def retrieve_pattern(self, new_state, sync, time = None):
        self.T = self.T_max
        self.state = new_state
        while self.T > 0.0001:
            self.update_state(sync)
            self.T *= self.T_decay
        return self.state


class Continuous_HN(Hopfield_network):
    def __init__(self, num_neurons, T,alpha, p):
        super().__init__(num_neurons, p)
        self.T = T
        self.alpha = alpha

    def update_state(self, sync=True):
        if sync == True:
            self.hidden_state = (np.dot(self.weights,self.state.reshape(-1,1))).flatten()
            self.state = sigmoid(-(1-self.alpha) * self.state + self.hidden_state, self.T)
        else:
            for i in range(self.num_neurons):
                self.hidden_state[i] = np.dot(self.weights[i,:], self.state)
                self.state[i] = sigmoid(-(1 - self.alpha) * self.state[i] + self.hidden_state[i], self.T)
        return None

if __name__ == '__main__':
    num_neurons = 100
    num_patterns = 12
    sync = False
    flips = 25
    HN = Hopfield_network(num_neurons=num_neurons, p=0.5)
    # HN = Continuous_HN(num_neurons=num_neurons, T=0.000001, alpha=1, p=0.5)
    # HN = Boltzmann_machine(num_neurons=num_neurons,T_max=0.1, T_decay=0.99, p=0.5)
    patterns = [random_state(0.5, num_neurons) for i in range(num_patterns)]
    HN.learn_patterns(patterns, rule='Storkey')

    pattern_r = introduce_random_flips(patterns[0], flips)

    print('Similarity with different patterns (before): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(pattern_r - patterns[i], 2))

    retrieved_pattern = np.sign(HN.retrieve_pattern(pattern_r, sync, 1000))
    print('Similarity with different patterns (after): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(retrieved_pattern - patterns[i], 2))
