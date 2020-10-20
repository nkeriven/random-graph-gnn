"""
@author: kerivenn
"""

from __future__ import division

import numpy as np

# plot
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as Fc

# My code
import util as ut

plt.close('all')

## CONVERGENCE: evaluation of invariant
np.random.seed(0)
torch.manual_seed(0)
ns = (10**np.linspace(2,3.5,10)).astype(int) # nodes
big_n = 5000
nexp = 30
size_layers = [1,10,10,10,10,10,1]
GN = ut.MyGCN(size_layers, variant='inv', nonlin=Fc.relu)

fz = lambda x,y:np.cos(10*x)
fsig = lambda x:1

# compute approximate limit value
output = np.zeros(nexp)
for e in range(nexp):
    print('Compute limit value {}/{}'.format(e+1, nexp))
    X,F = ut.surface_uniform(big_n, fz=fz)
    G = ut.random_graph_similarity(X,f=fsig,alpha=1, bandwidth=0.15, mode="Gaussian")
    D = ut.nx2tg(G, node_attr='node_attr')
    output[e] = GN(D.x, D.edge_index)

cont_value = output.mean()

# compute curve
output = np.zeros((nexp, len(ns), 3))
for e in range(nexp):
    for nind, n in enumerate(ns):
        for aind, alpha in enumerate([1, 1/n**(1/4), 1/n**(1/2)]):
            print('Evaluate convergence {}/{}, {}/{}, {}/{}'.format(aind+1, 3, nind+1, len(ns), e+1, nexp))
            X,F = ut.surface_uniform(n, fz=fz)
            G = ut.random_graph_similarity(X,f=fsig,alpha=alpha, bandwidth=0.15, mode="Gaussian")
            D = ut.nx2tg(G, node_attr='node_attr')
            output[e,nind,aind] = GN(D.x, D.edge_index)

output_ = np.abs(output-cont_value)
output_m = output_.mean(axis=0)
output_err = output_.std(axis=0)

# draw
plt.figure(figsize=(4,3))
marker=['-o', '-d', '-v']
for i in range(3):
    plt.plot(ns, output_m[:,i], marker[i], linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.legend([r'$\alpha = 1$',r'$\alpha = n^{-1/4}$',r'$\alpha = n^{-1/2}$'], fontsize=12)
plt.xlabel('n', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.show()
