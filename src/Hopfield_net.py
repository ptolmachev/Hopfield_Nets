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

class Hopfield_net():
    def __init__(self, num_neurons, T_max, T_decay, p):
        self.num_neurons = num_neurons
        self.weights = (1/self.num_neurons)*np.random.randn(self.num_neurons,self.num_neurons)
        self.state = np.array([random_activity(p) for i in range(self.num_neurons)]).reshape(1,-1)
        self.learned_patterns = None
        self.T = T_max
        self.T_max = T_max
        self.T_decay = T_decay

    def update_state(self):
        h = np.dot(self.weights,self.state.reshape(-1,1))
        self.state = np.array([(1 if np.random.rand() < h[i] else -1) for i in range(len(h))])
        return None

    def evolve(self):
        while self.T > 0.001:
            self.update_state()
            self.T *= self.T_decay
        return None

    def learn_patterns(self, patterns):
        patterns = np.array(patterns)
        self.weights = (1/self.num_neurons)*np.dot(patterns.T, patterns)
        self.learned_patterns = patterns
        return None

    def retrieve_pattern(self, new_state):
        self.T = self.T_max
        self.state = new_state
        while self.T > 0.001:
            self.update_state()
            self.T *= self.T_decay
        return self.state

    def set_weights(self, new_weights):
        self.weights = new_weights
        return None


if __name__ == '__main__':
    num_neurons = 100
    HN = Hopfield_net(num_neurons=num_neurons, T_max=1, T_decay = 0.999, p=0.5)

    # #np.array([random_activity(0.5) for i in range(num_neurons)])#
    pattern1 = np.array([(-1)**i for i in range(num_neurons)])
    pattern2 = np.hstack([-1.0*np.ones(num_neurons//2),np.ones(num_neurons//2)])
    pattern3 = np.hstack([((-1.0)**i)*np.ones(num_neurons//5) for i in range(5)])

    HN.learn_patterns([pattern1, pattern2])

    pattern_r = introduce_random_flips(pattern2, 10)

    print('Similarity with different patterns (before): ')
    print(np.linalg.norm(pattern_r - pattern1, 2))
    print(np.linalg.norm(pattern_r - pattern2, 2))
    print(np.linalg.norm(pattern_r - pattern3, 2))

    retrieved_pattern = HN.retrieve_pattern(pattern_r)

    print('Similarity with different patterns (after): ')
    print(np.linalg.norm(retrieved_pattern - pattern1, 2))
    print(np.linalg.norm(retrieved_pattern - pattern2, 2))
    print(np.linalg.norm(retrieved_pattern - pattern3, 2))