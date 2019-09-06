from src.Hopfield_net import Hopfield_network, sigmoid, random_state, introduce_random_flips
import numpy as np
from matplotlib import pyplot as plt

class Continuous_HN_oscillatory(Hopfield_network):
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
            self.biases = 1.*np.mean(patterns, axis = 0)

            Y = np.dot(patterns.T, patterns)
            # deprecate self-connectivity
            Y[np.arange(self.num_neurons),np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights = (1/self.num_neurons)*Y

            next_patterns = np.roll(patterns, axis = 0, shift = -1)
            Y = np.matmul(next_patterns.T, patterns)
            Y[np.arange(self.num_neurons), np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights += 0.02*(1 / self.num_neurons) * Y

        elif rule == 'Storkey':
            alpha = 0.98
            self.biases = 1.0 * np.mean(patterns, axis=0)
            next_patterns = np.roll(patterns, axis=0, shift=-1)
            for i in range(patterns.shape[0]):
                pattern = patterns[i]
                h = np.matmul(self.weights, pattern) - (np.diag(self.weights)*pattern) + self.biases
                H = np.hstack([h.reshape(-1,1) for j in range(self.num_neurons)]) - self.weights*pattern
                pattern_matrix = np.hstack([pattern.reshape(-1,1) for j in range(self.num_neurons)])
                Z = pattern_matrix - H
                self.weights += alpha*(1/self.num_neurons)*(Z*Z.T)

                next_pattern = next_patterns[i]
                h_next = np.matmul(self.weights, next_pattern) - (np.diag(self.weights)*next_pattern) + self.biases
                H_next = np.hstack([h_next.reshape(-1,1) for j in range(self.num_neurons)]) - self.weights*next_pattern
                pattern_matrix_next = np.hstack([next_pattern.reshape(-1,1) for j in range(self.num_neurons)])
                Z_next = pattern_matrix_next - H_next
                self.weights += (1-alpha)*(1/self.num_neurons)*(Z_next*Z.T)
        elif rule == 'associative':
            # self.biases = 1.0 * np.mean(patterns, axis=0)
            next_patterns = np.roll(patterns, axis=0, shift=-1)
            alpha = 0.9999
            self.weights += (1 / self.num_neurons) * np.dot(alpha*patterns.T + (1-alpha)*next_patterns.T, np.linalg.pinv(patterns.T))
        else:
            raise ValueError('the parameter rule can only take values \'Hebb\' or \'Storkey\'')
        return None

if __name__ == '__main__':
    num_neurons = 20
    num_patterns = 4
    sync = True
    rule = 'Storkey'
    flips = 0
    time = 500000
    HN = Continuous_HN_oscillatory(num_neurons=num_neurons, T=0.3, alpha=0.01, beta=0.000008, p=0.5)
    patterns = [random_state(0.5, num_neurons) for i in range(num_patterns)]
    # patterns = []
    # patterns.append(np.hstack([np.ones(8),np.zeros(4), np.zeros(4)]))
    # patterns.append(np.hstack([np.zeros(4),np.ones(8), np.zeros(4)]))
    # patterns.append(np.hstack([np.zeros(4), np.zeros(4), np.ones(8)]))
    HN.learn_patterns(patterns, rule=rule)

    pattern_r = introduce_random_flips(patterns[0], flips)

    print('\nSimilarity with different patterns (before): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(pattern_r - patterns[i], 2))

    retrieved_pattern, data = HN.retrieve_pattern(pattern_r, sync, time, record=True)
    print('\nSimilarity with different patterns (after): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(retrieved_pattern - patterns[i], 2))



    time_start = 200000
    fig1 = plt.figure()
    plt.plot(data['hidden_variables'].T[time_start:])
    plt.show()

    fig2 = plt.figure()
    plt.plot(data['state'].T[time_start:])
    plt.show()

    fig3 = plt.figure()
    similarities = np.array([[np.dot(data['state'][:,time_start:][:,j],patterns[i])/num_neurons for i in range(len(patterns))] for j in range(time-time_start)])
    plt.plot(similarities)
    plt.show()