
import numpy as np
import pickle
from src.Hopfield_net import *
from matplotlib import pyplot as plt

num_neurons = 100
num_of_flips = 100
num_of_patterns = 100
num_repetitions = 10
sync = True
time = 500
rule = 'Storkey'
sc = True
sc_or_not = '' if sc == False else '_sc'
file_name = f'flips_and_patterns_{rule}{sc_or_not}_{num_neurons}.pkl'
results = np.zeros((num_of_patterns, num_of_flips,num_repetitions))
for n_p in range(1,num_of_patterns+1):
    print(f'\n Testing memorising {n_p} patterns')
    for n_f in range(1,num_of_flips+1):
        print(f'Introducing {n_f} flips')
        for r in range(num_repetitions):
            HN = Hopfield_network(num_neurons=num_neurons)
            patterns = [random_state(0.5, num_neurons) for i in range(n_p)]
            HN.learn_patterns(patterns, rule=rule, sc=sc)
            pattern_r = introduce_random_flips(patterns[0], n_f)
            retrieved_pattern = HN.retrieve_pattern(pattern_r, sync, time, record=False)
            overlap = (1/num_neurons)*np.dot(retrieved_pattern,patterns[0])
            # print(overlap)
            results[n_p-1,n_f-1,r] = overlap
    if (n_p % 10 == 0):
        pickle.dump(results, open(file_name,'wb+'))


results = pickle.load(open(file_name,'rb+'))
results = (results[:,:results.shape[1]//2,:]-results[:,results.shape[1]//2:,:][:,::-1,:])/2
avg = np.mean(results, axis = -1)

fig, axs = plt.subplots(1, 1)
cs = axs.contourf(avg, levels = np.linspace(0,1,20), cmap = 'coolwarm', extend = 'both', linestyles = 'solid')
axs.contour(avg, levels = np.linspace(0,1,20), colors = 'k', linestyles = 'dashed', linewidths = 0.25)
axs.set_xlabel('Number of flips in the initial state')
axs.set_ylabel('Number of memorised patterns')
axs.set_title(f'The dependence of memory retrieval on flips in initial conditions \n and number of stored patterns \n (rule = {rule}, sc = {sc})')
fig.colorbar(cs)
plt.show()
fig.savefig(file_name.split('.pkl')[0] + '.png')

# plt.contour(np.arange(1,num_of_patterns+1),range(1,num_of_flips+1), avg, (0,), levels = np.linspace(0,1,50),extend='both')
# plt.show()
