"""
@author: kerivenn
"""

import numpy as np

# plot
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as Fc

# My code
from utils import surface_uniform, random_graph_similarity, nx2tg, my_draw3d
from models import MyGCN

plt.close('all')

torch.manual_seed(1234)
np.random.seed(1234)
n = 2000
fz=lambda x,y:0
fsig = lambda x:np.cos(9*x[1])
size_layers = [1,10,10,1]
GN = MyGCN(size_layers, variant='inv', nonlin=torch.sigmoid)
taus = np.linspace(0,10,5)
nexp = 20

# base value
output = np.zeros(nexp)
for e in range(nexp):
    print('Compute limit value {}/{}'.format(e+1, nexp))
    X,F = surface_uniform(n, fz=fz)
    G = random_graph_similarity(X,f=fsig,alpha=1, bandwidth=0.15, mode="Gaussian")
    D = nx2tg(G, node_attr='node_attr')
    output[e] = GN(D.x, D.edge_index)

cont_value = output.mean()

# stability
def t(x,tau):
    if x<1/3: return 0
    if x<2/3: return 3*tau*(x-1/3)
    return tau
outputs = np.zeros((nexp,len(taus)))
for e in range(nexp):
    for _,tau in enumerate(taus):
        print('Evaluate stability {}/{}, {}/{}'.format(_+1, len(taus), e+1, nexp))
        XX,F = surface_uniform(n, fz=fz)
        XX[:,2] = [t(x,tau) for x in XX[:,0]]
        Gp = random_graph_similarity(XX,f=fsig,alpha=1, bandwidth=0.15, mode="Gaussian")
        Dp = nx2tg(Gp, node_attr='node_attr')
        outputs[e,_] = GN(Dp.x, Dp.edge_index)

output_ = np.abs(outputs-cont_value)
output_m = output_.mean(axis=0)
output_err = output_.std(axis=0)

plt.figure(figsize=(4,3))
plt.errorbar(taus, output_m, yerr=output_err, linewidth=2)
plt.xlabel(r'$\|\nabla \tau\|_\infty$', fontsize=16)
plt.ylabel('Difference', fontsize=16)
plt.show()
