# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.optim as optim
from torch_geometric.utils import dense_to_sparse

from models import MyGWNN
from utils import random_graph_similarity, nx2tg, my_draw

from scipy import stats
from scipy.integrate import quad

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.close('all')

#%% parameters

# random graph
n = 150 # size of train graphs
nt = 400 # size of test graphs
n_sample = 5
mode = 'Gaussian' # kernel
sigma = .3
alpha = lambda n:min(1,3/n**(1/3)) # sparsity
random_edges = True # if false, true kernel. HUGE IN MEMORY !

# GNN
d0, d1 = [1,50,50,50], [50,50,1] # layer size before/after averaging
init = 1 if not random_edges else 2 # input order of filtering
n_epochs = 600

# plot
save = True

#%% utils

# Use this to change the density of x in [-1, 1]
# must be centered for thm to apply
def pdf(x, symmetric=False):
    if symmetric:
        return 1/2
    if x<0:
        return (2/7)*(1+x)
    else:
        return (5/7)*x

# function to approximate
# symmetric or not
def f(x, symmetric=False):
    if symmetric:
        return torch.Tensor(np.cos(5*x))
    return torch.Tensor(np.sin(5*x))

def generate_data(X):
    G, W = random_graph_similarity(X, bandwidth=sigma, return_expected=True,
                                         mode=mode, alpha=alpha(X.shape[0]))
    if not random_edges:
        ei, ew = dense_to_sparse(torch.Tensor(W)) # edge index, edge weights
    else:
        ei, ew = nx2tg(G).edge_index, None
    ei = ei.to(device)
    if not random_edges:
        ew = ew.to(device)
    return G, ei, ew


#%%

for (dist_sym, fun_sym) in [(1, 1), (1, 0), (0, 0)]:

    # latent variable distribution
    class my_distribution(stats.rv_continuous):        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
            # integrate area of the PDF in range a..b
            self.scale, _ = quad(lambda x: pdf(x,dist_sym), self.a, self.b)
    
        def _pdf(self, x):
            # scale PDF so that it integrates to 1 in range a..b
            return pdf(x,dist_sym) / self.scale  

    distribution = my_distribution(a=-1, b=1)
    
    data=[]
    for s in range(n_sample):
        X = distribution.rvs(size=n)[:,None]
        y = f(X, fun_sym)
        y = y.to(device)
        G, ei, ew = generate_data(X)
        G_plot = (G, ei, ew, X, y)
        data.append((ei, ew, y))
    
    # test
    Xt = distribution.rvs(size=nt)[:,None]
    yt = f(Xt, fun_sym)
    yt = yt.to(device)
    Gt, eit, ewt = generate_data(Xt)
    
    # GWNN
    GN = MyGWNN(d0, d1, normalization='sparsity', init=init)
    GN.to(device)
    
    # training
    optimizer = optim.Adam(GN.parameters(), lr=3e-4)
    for _ in range(n_epochs):
        loss_print = 0
        for (ei, ew, y) in data:
            optimizer.zero_grad()
            yhat = GN(n, ei, ew, sparsity=alpha(n), device=device)
            loss = ((y-yhat)**2).sum()/n
            loss.backward()
            optimizer.step()
            loss_print += loss.item()/n_sample
        random.shuffle(data)
        
        if np.mod(_,10) == 0:
            with torch.no_grad():
                yhatt = GN(nt, eit, ewt, sparsity=alpha(nt), device=device)
                losst = ((yt-yhatt)**2).sum()/nt
                print(f'Iter {_}, Train {loss_print}, Test {losst.item()}')
    
    yhat = GN(n, G_plot[1], G_plot[2], sparsity=alpha(n), device=device)
    
    yhat = yhat.cpu().detach()
    yhatt = yhatt.cpu().detach()
    
    plt.figure(figsize=(12,8))
    ax = np.linspace(-1,1,100)
    my_draw(Gt, pos=np.concatenate((Xt,yhatt), axis=1),
                  node_color=yhatt, node_size=60)
    plt.plot(ax, f(ax, fun_sym), c='r', linewidth=4)
    plt.subplots_adjust(wspace=0, hspace=0)
    if save:
        plt.savefig(f'fig/eq_radial{dist_sym}{fun_sym}.png', bbox_inches='tight',
                    transparent=True)
