import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

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
        self.linear = nn.Linear(args.embedding_dim, args.dim)
        nn_init(self.linear, 'xavier')
        self.args.eucl_vars.append(self.linear)	

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
        if self.args.select_manifold in {"poincare", "euclidean"}:
            return self.activation(node_repr)
        elif self.args.select_manifold == "lorentz":
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

    def hyperbolic_mean(self, y, node_num, max_neighbor, real_node_num, weight, dim=0, c=1.0, ):
        '''
        y [node_num * max_neighbor, dim]
        '''
        x = y[0:real_node_num*max_neighbor, :]
        weight_tmp = weight.view(-1,1)[0:real_node_num*max_neighbor, :]
        x = h2k(x)
        
        lamb = self.lorenz_factor(x, c=c, keepdim=True)
        lamb = lamb  * weight_tmp 
        lamb = lamb.view(real_node_num, max_neighbor, -1)

        x = x.view(real_node_num, max_neighbor, -1) 
        k_mean = (torch.sum(lamb * x, dim=1, keepdim=True) / (torch.sum(lamb, dim=1, keepdim=True))).squeeze()
        h_mean = k2h(k_mean)

        virtual_mean = torch.cat((torch.tensor([[1.0]]), torch.zeros(1,y.size(-1)-1)), 1).to(self.args.device)
        tmp = virtual_mean.repeat(node_num-real_node_num, 1)

        mean = torch.cat((h_mean, tmp), 0)
        return mean

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

    def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, mask):
        """
        message passing for a specific message type.
        """
        # pdb.set_trace()

        node_num, max_neighbor = adj_mat.shape[0], adj_mat.shape[1] 
        combined_msg = node_repr.clone()

        msg = torch.mm(node_repr, layer_weight) * mask
        real_node_num = (mask>0).sum()
        
        # select out the neighbors of each node
        neighbors = torch.index_select(msg, 0, adj_mat.view(-1))  # 这一步会产生很大的矩阵

        combined_msg = self.hyperbolic_mean(neighbors, node_num, max_neighbor, real_node_num, weight)
        return combined_msg 

    def get_combined_msg(self, step, node_repr, adj_mat, weight, mask):
        """
        perform message passing in the tangent space of x'
        """
        gnn_layer = 0 if self.args.tie_weight else step
        layer_weight = self.retrieve_params(self.msg_weight, gnn_layer)
        aggregated_msg = self.aggregate_msg(node_repr,
                                            adj_mat,
                                            weight,
                                            layer_weight, mask)
        combined_msg = aggregated_msg 
        return combined_msg


    def encode(self, node_repr, adj, weight):
        """
        
        """
        # pdb.set_trace()
        node_repr = self.activation(self.linear(node_repr))
        
        mask = torch.ones((node_repr.size(0),1)).to(self.args.device)
        node_repr = self.args.manifold.exp_map_zero(node_repr)

        for step in range(self.args.num_layers):
            node_repr = node_repr * mask
            tmp = node_repr
            combined_msg = self.get_combined_msg(step, node_repr, adj, weight, mask)
            combined_msg = (combined_msg) * mask
            node_repr = combined_msg * mask
            node_repr = self.apply_activation(node_repr) * mask
            real_node_num = (mask>0).sum()
            node_repr = self.args.manifold.normalize(node_repr)
        return node_repr