"""
@author: kerivenn
"""

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

def temp_draw(G, X, color, vmin=None, vmax=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pos = {i:X[i,:] for i in G.nodes}
    my_draw3d(G, pos=pos, node_color = color, alpha_edge=.7,
              ax=ax, display_axis=False, alpha=1, edge_color='k',
              vmin=vmin, vmax=vmax)
    ax.view_init(elev=45, azim=-105)

torch.manual_seed(1234)
np.random.seed(1234)
n = 2000 # nodes
fz=lambda x,y:0
fsig = lambda x:np.cos(4*x[1])
size_layers = [1,10,10,1]
GN = MyGCN(size_layers, variant='equi', nonlin=Fc.relu)

# base value
X,F = surface_uniform(n, fz=fz)
G = random_graph_similarity(X,f=fsig,alpha=.5, bandwidth=0.1, mode="Gaussian")
D = nx2tg(G, node_attr='node_attr')
base_val = GN(D.x, D.edge_index).flatten().detach().numpy()
temp_draw(G,X,base_val)

# stability
def t(x,tau):
    if x<1/3: return 0
    if x<2/3: return 3*tau*(x-1/3)
    return tau
for _,tau in enumerate([3,0]):
    XX = X
    XX[:,2] = [t(x,tau) for x in XX[:,0]]
    Gp = random_graph_similarity(XX,f=fsig,alpha=.5, bandwidth=0.1, mode="Gaussian")
    Dp = nx2tg(Gp, node_attr='node_attr')
    diff = np.abs(base_val - GN(Dp.x, Dp.edge_index).flatten().detach().numpy())
    if _==0: vM=np.percentile(diff,97)
    temp_draw(Gp,XX,diff,vmax=vM)
plt.show()
