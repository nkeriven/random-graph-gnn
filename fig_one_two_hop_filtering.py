# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import utils

import networkx as nx

plt.close('all')
np.random.seed(0)

#%% parameters

ns = np.logspace(1, 3, 10).astype(int)
n_sample = 50
mode = 'Gaussian' # kernel
sigma = .2
alpha = lambda n:3*n**(-1/3) # sparsity

# plot
save = True

# we consider simple GWNNs here, numpy-only
def simple_gwnn(A, init=1, nonlin='relu'):
    # A already divided by sparsity
    A = A.copy()
    if init==2:
        A = 3*A@A/A.shape[0] # adjust mult. constant, nicer for plotting
    A -= 1/4 # bias
    if nonlin == 'relu':
        A[A<0] = 0 # relu

    return np.mean(A, axis=0) # pooling, output is equivariant


#%%
plt.figure(figsize=(5,4))
for (init, label) in [(1, 'One-hop'),
                      (2, 'Two-hop')]:
    output=np.zeros((len(ns), n_sample))
    for n_ind, n in enumerate(ns):
        for _ in range(n_sample):
            print(f'{init}/2, {n_ind+1}/{len(ns)}, {_+1}/{n_sample}')
            X = (2*np.random.rand(n)-1)[:, None]
            G, W = utils.random_graph_similarity(X, bandwidth=sigma,
                                                 return_expected=True,
                                                 mode=mode, alpha=alpha(n))
            ytrue = simple_gwnn(W/alpha(n), init=init)
            y = simple_gwnn(nx.to_numpy_array(G)/alpha(n), init=init)
            output[n_ind, _] = ((ytrue-y)**2).sum()/n

    line, = plt.loglog(ns, np.mean(output, axis=1), label=label, linewidth=2)
    plt.fill_between(ns, np.percentile(output, 20, axis=1),
                        np.percentile(output, 80, axis=1),
                        color=line.get_color(), alpha=0.3)

plt.loglog(ns, 1/(100*np.sqrt(ns)*alpha(ns)), label='Theory')
plt.legend(fontsize=14)
plt.xlabel('n', fontsize=14)
plt.ylabel('MSE', fontsize=14)
if save:
    plt.savefig('fig/conv.pdf', bbox_inches='tight',
                transparent=True)