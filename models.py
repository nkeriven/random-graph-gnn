# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, get_laplacian, to_dense_adj

from torch_scatter import scatter_mean

#%% GCNs

class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, mode='GCN'):
        super(MyGCNConv, self).__init__(aggr='add', node_dim=0)  # "Add" aggregation (Step 5).
        self.lin_neigh = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin_in = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.constant_(self.bias, val=0)
        self.mode = mode
        
    def forward(self, x, edge_index, edge_weight=None, norm=1):
        # x has shape [N, in_channels] or [N, N, in_channels] for GWNN
        # edge_index has shape [2, E]
        T = self.lin_in(x) + self.propagate(edge_index, x=self.lin_neigh(x),
                                            norm=norm, edge_weight=edge_weight)
        if self.mode == 'GWNN':
            return T + self.bias[None, None, :]
        return  T + self.bias[None,:]

    def message(self, x_j, edge_weight=None, norm=1):
        # x_j: [E, out_channels] if GCN, [E, N, out_channels] if GWNN. (big!)
        # norm: [E,] if self.normalization=laplacian, scalar otherwise
        # edge_weight: [E,]
        if edge_weight is not None:
            ew = norm * edge_weight
            if self.mode=="GWNN":
                return x_j * ew[:,None,None]
            else:
                return x_j * ew[:,None]
        
        if np.isscalar(norm):
            return norm * x_j
        elif self.mode == "GWNN":
            return x_j * norm[:,None,None]
        else:
            return x_j * norm[:,None]


class MyGCN(nn.Module):
    def __init__(self, dims, nonlin=nn.functional.relu, variant='equi',
                 normalization='laplacian', mode='GCN'):
        super(MyGCN, self).__init__()
        self.dims = dims
        self.num_layers = len(dims)-2 # number of hidden layers
        self.variant = variant
        self.normalization = normalization
        self.mode = mode
        
        self.nonlin = nonlin
        
        self.layers = nn.ModuleList([MyGCNConv(dims[i], dims[i+1], mode=mode)\
                                      for i in range(self.num_layers)])
        self.output_lin = torch.nn.Linear(dims[-2], dims[-1], bias=True)
    
    def forward(self, x, edge_index, edge_weight=None,
                batch_vector=None, sparsity=1):
        # x has shape [N, in_channels] or [N, N, in_channels] for GWNN
        # Compute normalization.
        if self.normalization == 'laplacian':
            edge_id, norm = get_laplacian(edge_index, edge_weight,
                                          normalization='sym',
                                          num_nodes=x.shape[0])
            _, norm = remove_self_loops(edge_id, norm) # remove Id
            norm = -norm # shape [E,]
        elif self.normalization == 'sparsity':
            norm = 1/(x.shape[0]*sparsity) # scalar
        
        for i in range(self.num_layers):
            x = self.nonlin(self.layers[i].forward(x, edge_index, norm=norm,
                                                   edge_weight=edge_weight))

        x = self.output_lin(x)
        if self.variant == 'equi':
            return x
        else:
            if batch_vector is None:
                return x.mean(axis=0)
            else:
                return scatter_mean(x, batch_vector, dim=0)

class MyGWNN(nn.Module):
    def __init__(self, dims_before, dims_after, nonlin = nn.functional.relu,
                 normalization='laplacian', variant='equi',
                 init=2, device="cpu"):
        super(MyGWNN, self).__init__()
        self.dims_before = dims_before
        self.dims_after = dims_after
        self.variant = variant
        self.nonlin = nonlin
        self.init = init
        self.normalization = normalization
        
        self.GNNs = torch.nn.ModuleList([MyGCN(dims_before, nonlin=nonlin, variant='equi',
                                               mode='GWNN', normalization=normalization),
                                         MyGCN(dims_after, nonlin=nonlin, variant=variant,
                                               normalization=normalization)])
        
    def compute_init(self, n, edge_index, edge_weight=None, sparsity=1,
                     device='cpu'):
        
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).to(device)
        
        if self.normalization == 'laplacian':
            edge_id, edge_weight = get_laplacian(edge_index, edge_weight,
                                                 normalization='sym',
                                                 num_nodes=n)
            _, edge_weight = remove_self_loops(edge_id, edge_weight) # remove Id
            edge_weight = -edge_weight
            edge_weight = edge_weight.to(device)
        elif self.normalization == 'sparsity':
            edge_weight = edge_weight/(n*sparsity)
        
        # to tensor
        M = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0,:,:]
        output = n*M # A/sparsity or W at this point
        for _ in range(self.init-1):
            output = M @ output
        
        return output[:,:,None]
    
    def forward(self, n, edge_index, edge_weight=None,
                batch_vector=None, sparsity=1, device='cpu'):
        
        x = self.compute_init(n, edge_index, edge_weight=edge_weight,
                              sparsity=sparsity, device=device)
        
        x = self.GNNs[0](x, edge_index, edge_weight, sparsity=sparsity)
        x = x.sum(axis=1)/n
        x = self.GNNs[1](x, edge_index, edge_weight, batch_vector, sparsity)
        
        return x

