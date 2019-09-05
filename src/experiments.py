from src.Hopfield_net import Hopfield_network, sigmoid, random_state, introduce_random_flips
import numpy as np
from matplotlib import pyplot as plt

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
                self.u += self.beta * ( (self.hidden_state[i]+1) - self.u[i])
                self.state[i] = 2*sigmoid(self.hidden_state[i], self.T) - 1
        self.fr = (self.state + 1) / 2
        return None

    def learn_patterns(self, patterns, rule='Hebb'):
        patterns = np.array(patterns).squeeze()
        if rule == 'Hebb':
            Y = np.dot(patterns.T, patterns)
            # deprecate self-connectivity
            Y[np.arange(self.num_neurons),np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights = (1/self.num_neurons)*Y

            next_patterns = np.roll(patterns, axis = 0, shift = -1)
            Y = np.matmul(next_patterns.T, patterns)
            Y[np.arange(self.num_neurons), np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights += 0.005*(1 / self.num_neurons) * Y
        else:
            raise ValueError('the parameter rule can only take values \'Hebb\' or \'Storkey\'')
        return None

if __name__ == '__main__':
    num_neurons = 10
    num_patterns = 2
    sync = True
    rule = 'Hebb'
    flips = 0
    time = 1000
    HN = Continuous_HN_adaptive(num_neurons=num_neurons, T=0.3, alpha=1, beta=0.01, p=0.5)
    patterns = [random_state(0.5, num_neurons) for i in range(num_patterns)]
    HN.learn_patterns(patterns, rule=rule)

    pattern_r = introduce_random_flips(patterns[1], flips)

    print('\nSimilarity with different patterns (before): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(pattern_r - patterns[i], 2))

    retrieved_pattern, data = HN.retrieve_pattern(pattern_r, sync, time, record=True)
    print('\nSimilarity with different patterns (after): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(retrieved_pattern - patterns[i], 2))

    fig1 = plt.figure()
    plt.plot(data['hidden_variables'].T)
    plt.show()

    fig2 = plt.figure()
    plt.plot(data['state'].T)
    plt.show()

    fig3 = plt.figure()
    similarities = np.array([[np.linalg.norm(data['state'][:,j] - patterns[i],2) for i in range(len(patterns))] for j in range(time)])
    plt.plot(similarities)
    plt.show()