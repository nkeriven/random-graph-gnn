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

# My Code
from utils import surface_uniform, random_graph_similarity, nx2tg, my_draw3d
from models import MyGCN

plt.close('all')

## CONVERGENCE: evaluation of invariant
np.random.seed(0)
torch.manual_seed(0)
ns = [100, 500, 2000] # nodes
size_layers = [1,10,10,10,10,10,1]
GN = MyGCN(size_layers, variant='equi', nonlin=Fc.relu)

fz = lambda x,y:np.cos(10*x)
fsig = lambda x:1

# compute approximate limit value

for nind, n in enumerate(ns):
    X,F = surface_uniform(n, fz=fz)
    G = random_graph_similarity(X,f=fsig,alpha=10/n**(1/2), bandwidth=.4, mode="epsilon_graph")
    D = nx2tg(G, node_attr='node_attr')
    output = GN(D.x, D.edge_index).detach().numpy()
    
    pos = {i:X[i,:] for i in G.nodes}
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    my_draw3d(G, pos=pos, node_color = output,
              alpha_edge=.7, ax=ax, display_axis=False, alpha=1, edge_color='k')
    ax.view_init(elev=40, azim=-80)
plt.show()
