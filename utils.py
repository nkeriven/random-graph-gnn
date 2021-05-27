# -*- coding: utf-8 -*-


import numpy as np
import networkx as nx
import torch
from numba import jit
from torch_geometric.data import Data
import torch_geometric.utils as tgut
from nx_pylab3d import draw3d
from matplotlib.tri import Triangulation

from scipy.spatial.distance import pdist, squareform

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
        if len(X.shape)==1:
            X = X.view(-1,1)
        D.x = X
    
    return D

#%% plot

def my_draw(G,
            node_size=20,
            node_color='b',
            width=.1,
            edge_color='gray',
            pos=None,
            vmin=None, vmax=None, **kwds):
    if type(pos) == str:
        poss = [G.nodes[i][pos] for i in G.nodes]
        pos=poss
    nx.draw(G, node_size=node_size, node_color=node_color, width=width, 
            edge_color=edge_color, pos = pos,
            vmin=vmin, vmax=vmax, **kwds)
def my_draw3d(G,
              node_size=20,
              node_color='b',
              width=.1,
              edge_color='gray',
              pos=None,
              vmin=None, vmax=None, **kwds):
    if type(pos) == str:
        poss = [G.nodes[i][pos] for i in G.nodes]
        pos=poss
    draw3d(G, node_size=node_size, node_color=node_color, width=width, 
           edge_color=edge_color, pos = pos,
           vmin=vmin, vmax=vmax, **kwds)

#%% random graphs

def random_graph_similarity(X, f=None, alpha=1, mode="Gaussian", bandwidth=1,
                            warp = lambda X:X, return_expected = False):
    """RG model with similarity kernel that only depend on the distance 
    between latent variable.
    
    Input
        X           : n*d latent variables, 
                        or n*n pairwise distance matrix when mode=custom
        f           : signal over nodes that depends on latent variables.
        alpha       : sparsity level. Default 1
        mode        : "Gaussian" or "epsilon_graph"
        bandwidth   : bandwidth of kernel default 1
        warp        : modify X to generate non-TI kernels
    Output
        G       : NetworkX Graph
        G, W    : Nx Graph, n*n array (if return_expected=True)
    """
    
    n = X.shape[0]
    G = nx.empty_graph(n)
    
    if n>5000:
        print(f'Generate random graph with {n} nodes... (can be long)')
    
    if mode == "custom":
        edgelist = generate_edges(alpha*X)
        G.add_edges_from(edgelist)
        return G
    
    for i in range(n):
        G.nodes[i]['latent'] = X[i,:]
        if f is not None:
            G.nodes[i]['node_attr'] = f(X[i,:])
    
    # warp the kernel
    XX = warp(X)
    # generate edges
    if mode == "Gaussian":
        edgelist = generate_edges_gaussian(XX,bandwidth,alpha)
        G.add_edges_from(edgelist)
        if return_expected:
            W = alpha * np.exp(-squareform(pdist(XX, 'sqeuclidean'))
                               /(2*bandwidth**2))
            np.fill_diagonal(W,0)
    elif mode == "epsilon_graph":
        edgelist = generate_edges_epsilon_graph(XX,bandwidth,alpha)
        G.add_edges_from(edgelist)
        if return_expected:
            W = alpha * (squareform(pdist(XX, 'sqeuclidean'))<bandwidth**2)
            np.fill_diagonal(W,0)
    if f is None:
        for i in range(n):
            G.nodes[i]['node_attr'] = G.degree[i]
    
    if return_expected:
        return G, W
    return G

def random_community_size(K, alpha_dirichlet = 10):
    return np.random.dirichlet(alpha_dirichlet*np.ones(K))

def SBM(n, P, W, sort=False, return_expected=False, alpha=1):
    """Return a Stochastic Block model graph

    Input
        n   : Number of nodes
        P   : (K,) community sizes
        W   : (K,K) community connectivity
    Output
        G : NetworkX Graph
          (undirected) sbm graph. 
          Nodes communities are stored in G.nodes[i]['community'].
    G, E  : Nx Graph, expected values between communities
    """
    G = nx.empty_graph(n)
    K = P.shape[0]
    communities = np.array(np.random.choice(range(K), n, p=P))
    
    if sort:
        communities = np.sort(communities)
    for i in range(n):
        G.nodes[i]['community'] = communities[i]
    
    edgelist = generate_community_edges(communities, alpha*W)
    G.add_edges_from(edgelist)
    if return_expected:
        commx, commy = np.meshgrid(communities, communities)
        exp = alpha*W[commx, commy]
        return G, exp
    return G

def surface_uniform(n, fz = None):
    if fz is None:
        fz = lambda x,y: 0
    pos = np.zeros((n,3))
    pos[:,0], pos[:,1] = np.random.rand(n), np.random.rand(n)
    pos[:,2] = fz(pos[:,0], pos[:,1])
    tri = Triangulation(pos[:, 0], pos[:, 1])
    return pos, tri.triangles

@jit(nopython=True)
def generate_edges(C):
    ret = []
    for i in range(C.shape[0]):
        for j in range(i):
            if np.random.rand() < C[i,j]:
                ret.append((i,j))
    return ret

@jit
def generate_community_edges(communities, connectivity):
    ret = []
    for i in range(len(communities)):
        for j in range(i):
            if np.random.rand() < connectivity[communities[i],communities[j]]:
                ret.append((i,j))
    return ret

@jit(nopython=True)
def generate_edges_gaussian(X,sigma,alpha):
    ret = []
    for i in range(X.shape[0]):
        for j in range(i):
            vi, vj = X[i,:], X[j,:]
            if np.random.rand() < alpha*np.exp(-((vi-vj)**2).sum()
                                               /(2*sigma**2)):
                ret.append((i,j))
    return ret

@jit(nopython=True)
def generate_edges_epsilon_graph(X,epsilon,alpha):
    ret = []
    for i in range(X.shape[0]):
        for j in range(i):
            vi, vj = X[i,:], X[j,:]
            if np.random.rand() < alpha*(((vi-vj)**2).sum()<epsilon**2):
                ret.append((i,j))
    return ret