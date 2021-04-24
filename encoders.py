import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch_sparse
from torch_scatter import scatter_add

from utils.math_utils import h2k, k2h


def nn_init(nn_module, method='orthogonal'):
    """
    Initialize a Sequential or Module object
    Args:
        nn_module: Sequential or Module
        method: initialization method
    """
    if method == 'none':
        return
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            # for a Sequential object, the param_name contains both id and param name
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)

def init_weight(weight, method):
    """
    Initialize parameters
    Args:
        weight: a Parameter object
        method: initialization method 
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


class H2HGCN(nn.Module):
    
    def __init__(self, args):
        super(H2HGCN, self).__init__()
        self.debug = False
        self.args = args
        self.set_up_params()
        self.activation = nn.SELU()
        # self.linear = nn.Linear(args.embedding_dim, args.dim)
        # nn_init(self.linear, 'xavier')
        # self.args.eucl_vars.append(self.linear)	

    def set_up_params(self):
        """
        create the GNN params for a specific msg type
        """
        msg_weight = []
        layer = self.args.num_layers if not self.args.tie_weight else 1
        for iii in range(layer):
            M = torch.zeros([self.args.dim-1, self.args.dim-1], requires_grad=True)
            init_weight(M, 'orthogonal')
            M = nn.Parameter(M)
            self.args.stie_vars.append(M)
            msg_weight.append(M)
        self.msg_weight = nn.ParameterList(msg_weight)

    def apply_activation(self, node_repr):
        """
        apply non-linearity for different manifolds
        """
        return self.args.manifold.from_poincare_to_lorentz(
            self.activation(self.args.manifold.from_lorentz_to_poincare(node_repr))
        )

    def lorenz_factor(self, x, *, c=1.0, dim=-1, keepdim=False):
        """
            Calculate Lorenz factors
        """
        x_norm = x.pow(2).sum(dim=dim, keepdim=keepdim)
        x_norm = torch.clamp(x_norm, 0, 0.9)
        tmp = 1 / torch.sqrt(1 - c * x_norm)
        return tmp

    def hyperbolic_mean(self, x, adj_train_norm):

        adj_train_norm = adj_train_norm.coalesce()
        edge_index = adj_train_norm.indices()
        edge_weight = adj_train_norm.values()
        x = h2k(x)
        lamb = self.lorenz_factor(x)
        n = lamb.shape[0]
        lamb_indices = torch.arange(n).repeat(2, 1).to(lamb.device)

        edge_index, edge_weight = torch_sparse.spspmm(edge_index, edge_weight, lamb_indices, lamb, n, n, n)
        edge_index, edge_weight = self.adj_norm(edge_index, edge_weight, n)
        
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(n, n))

        k_mean = adj @ x
        h_mean = k2h(k_mean)
        return h_mean
        


    def retrieve_params(self, weight, step):
        """
        Args:
            weight: a list of weights
            step: a certain layer
        """
        layer_weight = torch.cat((torch.zeros((self.args.dim-1, 1)).to(self.args.device), weight[step]), dim=1)
        tmp = torch.zeros((1, self.args.dim)).to(self.args.device)
        tmp[0,0] = 1
        layer_weight = torch.cat((tmp, layer_weight), dim=0)
        return layer_weight

    def aggregate_msg(self, node_repr, adj_train_norm, layer_weight):
        """
        message passing for a specific message type.
        """

        # msg = torch.mm(node_repr, layer_weight)
        msg = node_repr
        
        combined_msg = self.hyperbolic_mean(msg, adj_train_norm)
        return combined_msg 

    def get_combined_msg(self, step, node_repr, adj_train_norm):
        """
        perform message passing in the tangent space of x'
        """
        gnn_layer = 0 if self.args.tie_weight else step
        layer_weight = self.retrieve_params(self.msg_weight, gnn_layer)
        aggregated_msg = self.aggregate_msg(node_repr, adj_train_norm, layer_weight)
        combined_msg = aggregated_msg 
        return combined_msg


    def encode(self, node_repr, adj_train_norm):
        """
        
        """
        # node_repr = self.activation(self.linear(node_repr))
        
        # node_repr = self.args.manifold.exp_map_zero(node_repr)
        reprs = []
        for step in range(self.args.num_layers):
            combined_msg = self.get_combined_msg(step, node_repr, adj_train_norm)
            node_repr = combined_msg
            node_repr = self.apply_activation(node_repr) 
            node_repr = self.args.manifold.normalize(node_repr)
            reprs.append(node_repr)
        # return node_repr
        if self.args.res_sum:
            return sum(reprs) / len(reprs)
        else:
            return node_repr

    def adj_norm(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-1.0)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, edge_weight * deg_inv_sqrt[row]