"""
@author: kerivenn
"""

import numpy as np
import torch.nn as nn
import torch
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from numba import jit
from torch_scatter import scatter_mean
from matplotlib.tri import Triangulation
import torch_geometric.utils as tgut
from torch_geometric.data import Data
import nx_pylab3d as gdraw


#%% own simplified implementation of GCN with normalized Laplacian

class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MyGCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Linearly transform node feature matrix.
        x = self.lin(x)

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm) + self.bias[None,:]

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Normalize node features.
        return norm.view(-1, 1) * x_j


class MyGCN(nn.Module):
    def __init__(self, dims, nonlin = nn.functional.relu, variant = 'equi', order=2):
        super(MyGCN, self).__init__()
        self.dims = dims
        self.num_layers = len(dims)-2 # number of hidden layers
        self.variant = variant
        self.Ls = []
        
        self.nonlin = nonlin
        
        # inner layers
        self.layers = nn.ModuleList([MyGCNConv(dims[i], dims[i+1])\
                                      for i in range(self.num_layers)])
        self.output_weights = nn.Parameter(torch.Tensor(dims[-2], dims[-1]))
        self.output_bias = nn.Parameter(torch.Tensor(dims[-1]))
        
        nn.init.xavier_normal_(self.output_weights)
        nn.init.constant_(self.output_bias, val = 0)

    
    def forward(self, x, edge_index, batch_vector=None):
        if len(x.shape)==1:
            x = x[:,None]
        for i in range(self.num_layers):
            x = self.nonlin(self.layers[i].forward(x, edge_index))

        x = (x[:,:,None]*self.output_weights[None,:,:]).sum(axis=1) + self.output_bias[None,:]
        if self.variant == 'equi':
            return x
        else:
            if batch_vector is None:
                return x.mean(axis=0)
            else:
                return scatter_mean(x, batch_vector, dim=0)

#%% random graphs

def surface_uniform(n, fz = None):
    if fz is None:
        fz = lambda x,y: 0
    pos = np.zeros((n,3))
    pos[:,0], pos[:,1] = np.random.rand(n), np.random.rand(n)
    pos[:,2] = fz(pos[:,0], pos[:,1])
    tri = Triangulation(pos[:, 0], pos[:, 1])
    return pos, tri.triangles

def random_graph_similarity(X, f=None, alpha=1, mode="Gaussian", bandwidth=1):
    """RG model with similarity kernel that only depend on the distance between latent variable.
    
    Input
        X           : n*d latent variables, or n*n pairwise distance matrix when mode=custom
        f           : signal over nodes that depends on latent variables.
        alpha       : sparsity level. Default 1
        mode        : "Gaussian" or "epsilon_graph"
        bandwidth   : bandwidth of kernel default 1
    Output
        G       : NetworkX Graph
    """
    
    n = X.shape[0]
    G = nx.empty_graph(n)
    
    if n>10000:
        print('Generate random graph with {} nodes... (can be long)'.format(n))
    
    if mode == "custom":
        edgelist = generate_independent_edges(alpha*X)
        G.add_edges_from(edgelist)
        return G
    
    for i in range(n):
        G.nodes[i]['latent'] = X[i,:]
        if f is not None:
            G.nodes[i]['node_attr'] = f(X[i,:])
    
    if mode == "Gaussian":
        edgelist = generate_independent_edges_gaussian(X,bandwidth,alpha)
        G.add_edges_from(edgelist)
    elif mode == "epsilon_graph":
        edgelist = generate_independent_edges_epsilon_graph(X,bandwidth,alpha)
        G.add_edges_from(edgelist)
    
    if f is None:
        for i in range(n):
            G.nodes[i]['node_attr'] = G.degree[i]
    
    return G

@jit(nopython=True)
def generate_independent_edges(C):
    ret = []
    for i in range(C.shape[0]):
        for j in range(i):
            if np.random.rand() < C[i,j]:
                ret.append((i,j))
    return ret

@jit(nopython=True)
def generate_independent_edges_gaussian(X,sigma,alpha):
    ret = []
    for i in range(X.shape[0]):
        for j in range(i):
            vi, vj = X[i,:], X[j,:]
            if np.log(np.random.rand()) < np.log(alpha) - ((vi-vj)**2).sum()/(2*sigma**2):
                ret.append((i,j))
    return ret

@jit(nopython=True)
def generate_independent_edges_epsilon_graph(X,epsilon,alpha):
    ret = []
    for i in range(X.shape[0]):
        for j in range(i):
            vi, vj = X[i,:], X[j,:]
            if np.random.rand() < alpha*(((vi-vj)**2).sum()<epsilon**2):
                ret.append((i,j))
    return ret

#%% conversion utils

def nx2tg(G, node_attr = None, edge_attr = None, undirected = True):
    """
    Takes a networkx graph and return a signal matrix + edge list suited for torch geometric.        
    
    Parameters
    ----------
    G : networkx Graph.
    keyword : string, optional
        Key for the signal at each node: G.nodes[i][keyword]. The default is 'signal'.
    nosig : string, optional
        If keyword is None, computes a signal that depends only on the graph structure.
        The default is 'degree'.

    Returns
    -------
    X : (n * d) Matrix of d-dimensional signal at each node.
    E : list of edges

    """
    
    edgelist = torch.LongTensor([list(e) for e in G.edges]).t()
    if undirected:
        edgelist = tgut.to_undirected(edgelist)
    
    if edge_attr is not None:
        EA = [G.edges[tuple(e)][edge_attr] for e in edgelist.t().numpy()]
    else:
        EA = None
    
    D = Data(x=torch.ones(len(G)), edge_index=edgelist, edge_attr=EA)
    
    if node_attr is not None:
        X = torch.Tensor([G.nodes[i][node_attr] for i in G.nodes])
        D.x = X
    
    return D

#%% drawing

def my_draw3d(G, ax=None, node_size=20, node_color='b', width=.1, 
              edge_color='gray', alpha_edge=.5, **kwds):
    gdraw.draw3d(G, ax=ax, node_size=node_size, node_color=node_color, width=width, edge_color=edge_color, 
                 alpha_edge=alpha_edge, **kwds)
