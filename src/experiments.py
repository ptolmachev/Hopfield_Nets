from src.Hopfield_net import *
import numpy as np
from matplotlib import pyplot as plt

def pseudo_sum(Z):
    return np.hstack([np.sum(Z, axis=1).reshape(-1,1) for i in range(Z.shape[-1])]) - Z



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

    def learn_cycles(self, patterns, mix_coeff, rule='Hebb'):
        patterns = np.array(patterns).squeeze()
        if rule == 'Hebb':
            self.biases = 1.*np.mean(patterns, axis = 0)

            Y = np.dot(patterns.T, patterns)
            # deprecate self-connectivity
            Y[np.arange(self.num_neurons),np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights = mix_coeff*(1/self.num_neurons)*Y

            next_patterns = np.roll(patterns, axis = 0, shift = -1)
            Y = np.matmul(next_patterns.T, patterns)
            Y[np.arange(self.num_neurons), np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            self.weights += (1 - mix_coeff)*(1 / self.num_neurons) * Y

        elif rule == 'Storkey':
            self.biases = 1.0 * np.mean(patterns, axis=0)
            next_patterns = np.roll(patterns, axis=0, shift=-1)
            self.weights = np.zeros((self.num_neurons,self.num_neurons))
            for i in range(patterns.shape[0]):
                pattern = patterns[i]
                h = np.matmul(self.weights, pattern) - (np.diag(self.weights)*pattern) + self.biases
                H = np.hstack([h.reshape(-1,1) for j in range(self.num_neurons)]) - self.weights*pattern
                pattern_matrix = np.hstack([pattern.reshape(-1,1) for j in range(self.num_neurons)])
                Z = pattern_matrix - H
                self.weights += mix_coeff*(1/self.num_neurons)*(Z*Z.T)

                next_pattern = next_patterns[i]
                h_next = np.matmul(self.weights, next_pattern) - (np.diag(self.weights)*next_pattern) + self.biases
                H_next = np.hstack([h_next.reshape(-1,1) for j in range(self.num_neurons)]) - self.weights*next_pattern
                pattern_matrix_next = np.hstack([next_pattern.reshape(-1,1) for j in range(self.num_neurons)])
                Z_next = pattern_matrix_next - H_next
                self.weights += (1-mix_coeff)*(1/self.num_neurons)*(Z_next*Z.T)

        elif rule == 'projection_association':
            #the projection rule for oscillations works only with biases!
            self.biases = 1.0 * np.mean(patterns, axis=0)
            next_patterns = np.roll(patterns, axis=0, shift=-1)
            # Y = np.matmul(patterns.T, np.linalg.pinv(patterns.T))
            # # Y[np.arange(self.num_neurons), np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            # self.weights = Y

            Z = patterns.T
            Z_next = next_patterns.T

            # these two lines are the same
            # Y = np.matmul(Z, np.matmul(np.linalg.pinv(np.matmul(Z.T,Z)), Z.T))
            Y = np.matmul(Z, np.linalg.pinv(Z))

            #works for some yet unknown reason!
            Y_t = (np.matmul(Z_next, np.matmul(np.linalg.pinv(np.matmul(Z.T,Z_next)), Z.T)))

            #doesn't work properly
            # Y_t = np.matmul(Z_next, np.matmul(np.linalg.pinv(np.matmul(Z.T, Z)), Z.T))

            self.weights = mix_coeff*Y + (1 - mix_coeff)*Y_t

        elif rule == 'projection_experimental':
            self.biases = 1 * np.mean(patterns, axis=0)
            next_patterns = np.roll(patterns, axis=0, shift=-1)

            Z = patterns.T
            H = self.T * np.arctanh(Z) - np.hstack([self.biases.reshape(-1, 1) for i in range(Z.shape[1])])
            Y = (np.matmul(H, np.matmul(np.linalg.pinv(np.matmul(Z.T, H)), Z.T)))

            Z_next = next_patterns.T
            H_next = self.T * np.arctanh(Z_next) - np.hstack([self.biases.reshape(-1, 1) for i in range(Z.shape[1])])
            Y_t = (np.matmul(H_next, np.matmul(np.linalg.pinv(np.matmul(Z.T, H_next)), Z.T)))
            self.weights = mix_coeff*Y + (1 - mix_coeff)*Y_t

        elif rule == 'adaptive':
            self.biases = 1 * np.mean(patterns, axis=0)
            self.weights = np.zeros((self.num_neurons, self.num_neurons))
            Z = patterns.T
            Z_next = np.roll(Z, axis=1, shift=1)
            num_patterns = Z.shape[-1]
            for j in range(3):
                Gs = []
                for i in range(patterns.shape[0]):
                    pattern = patterns[i]
                    h = np.matmul(self.weights, pattern) - (np.diag(self.weights)*pattern) + self.biases
                    H = np.hstack([h.reshape(-1,1) for j in range(self.num_neurons)]) - self.weights*pattern
                    pattern_matrix = np.hstack([pattern.reshape(-1,1) for j in range(self.num_neurons)])
                    G = pattern_matrix - H
                    Gs.append(deepcopy(G))
                for i in range(patterns.shape[0]):
                    self.weights += (0.2205)*(1/self.num_neurons)*(Gs[i]*Gs[i].T)

            self.weights[np.arange(self.num_neurons), np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
            #
            c = 0.8
            A = (Z + np.ones(Z.shape))/2
            B = (Z + Z_next - c*np.minimum(Z_next,0))
            C = np.sum(np.array([np.matmul(A[:,i].reshape(-1,1), B[:,i].reshape(1,-1)) for i in range(A.shape[-1])]), axis = 0)
            self.weights += 0.0968*(1/self.num_neurons)*(np.matmul(A,Z.T) - C)
            self.weights[np.arange(self.num_neurons), np.arange(self.num_neurons)] = np.zeros(self.num_neurons)
        else:
            raise ValueError('the parameter rule can only take values \'Hebb\' or \'Storkey\'')
        return None

if __name__ == '__main__':
    num_neurons = 4
    num_patterns = 4
    sync = True

    flips = 0
    time = 200000
    # mix_coeff = 0.75
    # rule = 'projection'
    # HN = Continuous_HN_oscillatory(num_neurons=num_neurons, T=0.3, alpha=0.01, beta=0.000009, p=0.5)
    mix_coeff = 0.95
    rule = 'adaptive'
    HN = Continuous_HN_oscillatory(num_neurons=num_neurons, T=0.27, alpha=0.1, beta=0.0004, p=0.5)
    # patterns = [random_state(0.5, num_neurons) for i in range(num_patterns)]
    # patterns = []
    # patterns.append(np.hstack([0.999*np.ones(10),-0.999*np.ones(5), -0.999*np.ones(5)]))
    # patterns.append(np.hstack([-0.999*np.ones(5),0.999*np.ones(10), -0.999*np.ones(5)]))
    # patterns.append(np.hstack([-0.999*np.ones(5), -0.999*np.ones(5), 0.999*np.ones(10)]))

    patterns = []
    patterns.append(np.hstack([+0.999,+0.999, -0.999, -0.999]))
    patterns.append(np.hstack([-0.999,+0.999, +0.999, -0.999]))
    patterns.append(np.hstack([-0.999,-0.999, +0.999, +0.999]))
    patterns.append(np.hstack([+0.999,-0.999, -0.999, +0.999]))

    HN.learn_cycles(patterns, mix_coeff=mix_coeff, rule=rule)

    pattern_r = introduce_random_flips(patterns[0], flips)

    print('\nSimilarity with different patterns (before): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(pattern_r - patterns[i], 2))

    retrieved_pattern, data = HN.retrieve_pattern(pattern_r, sync, time, record=True)
    print('\nSimilarity with different patterns (after): ')
    for i in range(len(patterns)):
        print(np.linalg.norm(retrieved_pattern - patterns[i], 2))



    time_start = 500#200000
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