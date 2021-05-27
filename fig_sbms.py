# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.utils import dense_to_sparse

from models import MyGWNN, MyGCN
from utils import SBM, nx2tg, my_draw

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.close('all')


#%% parameters

# SBM with constant expected degree
K = 2
tau = 0.35 # community unbalance
P = np.array([tau, 1-tau])
gamma = 0.7 # separability
WW = np.array([[gamma, (1-gamma)*tau/(1-tau)],
              [(1-gamma)*tau/(1-tau), 
               (tau-(1-gamma)*tau**2/(1-tau))/(1-tau)]])

# random graph
n = 80 # size of train graphs
n_sample = 5
nt = 300 # size of test graphs
alpha = lambda n: 1 # sparsity
random_edges = True # if false, true kernel. HUGE IN MEMORY !

# GNN
d0, d1 = [1,50,50,50], [50,50,K] # dims of gwnn before/after pooling
dd = [1,250,250,250,250,250,K] # dims of gnn
init = 1 if not random_edges else 2 # input order of filtering
n_epochs = 1000

# plot
save = True
vmin, vmax = 0, 1

#%%

def generate_data(n):
    G, W = SBM(n, P, WW, return_expected=True, alpha=alpha(n))
    y = torch.Tensor([G.nodes[i]['community'] for i in G.nodes])
    if not random_edges:
        ei, ew = dense_to_sparse(torch.Tensor(W)) # edge index, edge weights
    else:
        ei, ew = nx2tg(G).edge_index, None
    y = y.to(device)
    ei = ei.to(device)
    if not random_edges:
        ew = ew.to(device)
    return G, ei, ew, y

# training random graph
data = []
for _ in range(n_sample):
    G, ei, ew, y = generate_data(n)
    G_plot = (G, ei, ew, y)
    data.append((ei, ew, y))

# test random graph
Gt, eit, ewt, yt = generate_data(nt)

# consistent plot layout
posG = nx.drawing.layout.spring_layout(G_plot[0])
posGt = nx.drawing.layout.spring_layout(Gt)

#%% GWNN

GN = MyGWNN(d0, d1, normalization='sparsity', init=init)
GN.to(device)

# training
optimizer = optim.Adam(GN.parameters(), lr=3e-4)
print('test GWNN...')
for _ in range(n_epochs):
    loss_print = 0
    loss=0
    for (ei, ew, y) in data:
        optimizer.zero_grad()
        yhat = nn.LogSoftmax()(GN(n, ei, ew, sparsity=alpha(n), device=device))
        loss += nn.NLLLoss()(yhat, y.long())
    loss.backward()
    optimizer.step()
    loss_print += loss.item()/n_sample
    random.shuffle(data)
    
    if np.mod(_,10) == 0:
        with torch.no_grad():
            yhatt = nn.LogSoftmax()(GN(nt, eit, ewt, sparsity=alpha(nt),
                                        device=device))
            losst = nn.NLLLoss()(yhatt, yt.long())
            print(f'Iter {_}, Train {loss_print}, Test {losst.item()}')

yhat = nn.LogSoftmax()(GN(n, G_plot[1], G_plot[2], sparsity=alpha(n),
                          device=device))

plt.figure(figsize=(16,6))
plt.subplot(121)
my_draw(G_plot[0], node_color=np.exp(yhat.cpu().detach().numpy())[:,0],
              pos=posG, width=.15, vmin=vmin, vmax=vmax, node_size = 80)
plt.title('GWNN, Train', fontsize=28)
plt.subplot(122)
my_draw(Gt, node_color=np.exp(yhatt.cpu().detach().numpy())[:,0],
              pos=posGt, width=.05, vmin=vmin, vmax=vmax, node_size = 80)
plt.title('GWNN, Test', fontsize=28)
plt.subplots_adjust(wspace=0, hspace=0)
if save:
    plt.savefig('fig/eq_sbm_gwnn_graph.png', bbox_inches='tight',
                transparent=True)

#%% GNN

GNN = MyGCN(dd, normalization='sparsity')
GNN.to(device)

# training
optimizer = optim.Adam(GNN.parameters(), lr=3e-4)
constant_input = (torch.ones(n,1)).to(device)
constant_inputt = (torch.ones(nt,1)).to(device)
print('test GNN...')
    
for _ in range(2*n_epochs):
    loss_print = 0
    loss=0
    for (ei, ew, y) in data:
        optimizer.zero_grad()
        yhat = nn.LogSoftmax()(GNN(constant_input, ei, ew, sparsity=alpha(n)))
        loss += nn.NLLLoss()(yhat, y.long())
    loss.backward()
    optimizer.step()
    loss_print += loss.item()/n_sample
    random.shuffle(data)
    
    if np.mod(_,10) == 0:
        with torch.no_grad():
            yhatt = nn.LogSoftmax()(GNN(constant_inputt, eit, ewt, sparsity=alpha(nt)))
            losst = nn.NLLLoss()(yhatt, yt.long())
            print(f'Iter {_}, Train {loss_print}, Test {losst.item()}')

yhat = nn.LogSoftmax()(GNN(constant_input, G_plot[1], G_plot[2], sparsity=alpha(n)))

plt.figure(figsize=(16,6))
plt.subplot(121)
my_draw(G_plot[0], node_color=np.exp(yhat.cpu().detach().numpy())[:,0],
              pos=posG, width=.15, vmin=vmin, vmax=vmax, node_size = 80)
plt.title('GNN, Train', fontsize=28)
plt.subplot(122)
my_draw(Gt, node_color=np.exp(yhatt.cpu().detach().numpy())[:,0],
              pos=posGt, width=.05, vmin=vmin, vmax=vmax, node_size = 80)
plt.title('GNN, Test', fontsize=28)
plt.subplots_adjust(wspace=0, hspace=0)
if save:
    plt.savefig('fig/eq_sbm_gnn_graph.png', bbox_inches='tight',
                transparent=True)