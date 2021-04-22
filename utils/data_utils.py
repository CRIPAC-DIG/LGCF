"""Data utils functions for pre-processing and data loading."""
import os
import pickle
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from utils.pre_utils import normalize_weight, pad_sequence
from utils.citation_utils import load_citation_data
import pdb
import torch_geometric.transforms as T


def load_data(args, datapath):
    processed_path = os.path.join(datapath, 'processed_data.pkl')
    if os.path.exists(processed_path):
        print(f'Loading processed data from {processed_path}')
        with open(processed_path, 'rb') as f:
            return pickle.load(f)
    
    if args.task == 'lp':
        data = load_data_lp(args, args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, hgnn_adj, hgnn_weight = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false 
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
            data['hgnn_adj'] = hgnn_adj
            data['hgnn_weight'] = hgnn_weight
    else:
        raise NotImplementedError

    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    with open(processed_path, 'wb') as f:
        print(f'Dumping processed data to {processed_path}')
        pickle.dump(data, f)
    return data

# ############### FEATURES PROCESSING ####################################

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ############### DATA SPLITS #####################################################

def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  
    x, y = sp.triu(adj).nonzero() # triu: Return the upper triangular portion of a matrix in sparse format
    pos_edges = np.array(list(zip(x, y)))  # (m, 2)
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero() 
    neg_edges = np.array(list(zip(x, y))) 
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop) 
    n_test = int(m_pos * test_prop) 
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]     
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    """
    根据 adj_train 得到一个权重为 1 的邻接矩阵.
    """
    tmp_a, tmp_b = adj_train.nonzero()
    hgnn_adj = [[i] for i in range(adj.shape[0])]
    hgnn_weight = [[1] for i in range(adj.shape[0])]
    indptr_tmp = adj_train.indptr
    indices_tmp = adj_train.indices
    data_tmp = adj_train.data
    flag = 0
    for i in range(len(indptr_tmp)-1):
        items = indptr_tmp[i+1] - indptr_tmp[i]
        for j in range(items):
            hgnn_adj[i].append(indices_tmp[flag])
            hgnn_weight[i].append(1)
            flag += 1

    max_len = max([len(i) for i in hgnn_adj])
    normalize_weight(hgnn_adj, hgnn_weight)

    # 用 0 补齐
    hgnn_adj = pad_sequence(hgnn_adj, max_len)
    hgnn_weight = pad_sequence(hgnn_weight, max_len)
    hgnn_adj = np.array(hgnn_adj)
    hgnn_weight = np.array(hgnn_weight)

    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false), torch.from_numpy(hgnn_adj).cuda(), torch.from_numpy(hgnn_weight).cuda().float()



# ############### LINK PREDICTION DATA LOADERS ####################################

def load_data_lp(args, dataset, use_feats, data_path):
    if dataset in ['disease_lp']:
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset in ('cora', 'pubmed', 'citeseer'):
        # adj, features = load_citation_data(dataset, data_path)[:2]
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features, }
    return data


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels
