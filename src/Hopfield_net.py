import numpy as np
from copy import deepcopy
def random_activity(p):
    return np.random.choice([-1, 1], 1, [1 - p, p])[0]

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
        self.biases = np.zeros(num_neurons)
        self.state = np.array([random_activity(p) for i in range(self.num_neurons)]).reshape(1,-1)

    def update_state(self, sync=True):
        if sync == True:
            h = (np.dot(self.weights,self.state.reshape(-1,1)).T + self.biases).flatten()
            self.state = np.sign(h)
        elif sync == False:
            for i in range(num_neurons):
                h_i = np.dot(self.weights[i,:], self.state)
                self.state[i] = np.sign(h_i)
        else:
            raise AttributeError('sync variable can take only boolean values')
        return None

    def learn_patterns(self, patterns):
        patterns = np.array(patterns)
        Y = np.dot(patterns.T, patterns)
        Y[np.arange(self.num_neurons),np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
        self.weights = (1/self.num_neurons)*np.dot(patterns.T, patterns)
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
            h = (np.dot(self.weights, self.state.reshape(-1, 1)).T + self.biases).flatten()
            self.state = np.array([(1 if np.random.rand() < h[i] else -1) for i in range(len(h))])
        elif sync == False:
            for i in range(num_neurons):
                h_i = np.dot(self.weights[i,:], self.state)
                self.state[i] = 1 if np.random.rand() < h_i else -1
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

if __name__ == '__main__':
    num_neurons = 100
    sync = False
    HN = Hopfield_network(num_neurons=num_neurons, p=0.5)
    # HN = Boltzmann_machine(num_neurons=num_neurons,T_max=10, T_decay=0.99, p=0.5)
    pattern1 = np.array([(-1)**i for i in range(num_neurons)])
    pattern2 = np.hstack([((-1)**i)*np.ones(num_neurons//50) for i in range(50)])
    pattern3 = np.hstack([-1.0*np.ones(num_neurons//2),np.ones(num_neurons//2)])
    HN.learn_patterns([pattern1, pattern2, pattern3])

    pattern_r = introduce_random_flips(pattern1, 10)

    print('Similarity with different patterns (before): ')
    print(np.linalg.norm(pattern_r - pattern1, 2))
    print(np.linalg.norm(pattern_r - pattern2, 2))
    print(np.linalg.norm(pattern_r - pattern3, 2))

    retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, 100)
    print('Similarity with different patterns (after): ')
    print(np.linalg.norm(retrieved_pattern - pattern1, 2))
    print(np.linalg.norm(retrieved_pattern - pattern2, 2))
    print(np.linalg.norm(retrieved_pattern - pattern3, 2))