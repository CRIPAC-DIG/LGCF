import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import pdb

from  manifolds import LorentzManifold
from encoders import H2HGCN


class LGCFModel(nn.Module):

    def __init__(self, users_items, args):
        super(LGCFModel, self).__init__()

        self.c = torch.tensor([args.c]).cuda()

        self.manifold = args.manifold = LorentzManifold(args)
        self.nnodes = args.n_nodes

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.args = args

        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(args.device)

        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.exp_map_zero(self.embedding.state_dict()['weight']))
        self.args.eucl_vars.append(self.embedding.weight)

        # args.dim = args.embedding_dim + 1
        args.dim = args.embedding_dim
        self.encoder = H2HGCN(args).cuda()
        

    def encode(self, adj_train_norm):
        x = self.embedding.weight
        h = self.encoder.encode(x, adj_train_norm)
        return h

    def decode(self, h, idx):
        # pdb.set_trace()
        # pdb.set_trace()
        if isinstance(h, list):
            sqdists = []
            for emb in h:
                emb_in = emb[idx[:, 0], :]
                emb_out = emb[idx[:, 1], :]
                sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
                sqdists.append(sqdist)
            return sum(sqdists) / len(sqdists)
        else:
            emb_in = h[idx[:, 0], :]
            emb_out = h[idx[:, 1], :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            return sqdist


    def compute_loss(self, embeddings, triples):
        # pdb.set_trace()
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]]
                                    for i in range(self.args.num_neg)]

        pos_scores = self.decode(embeddings, train_edges)

        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in
                           sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)

        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            if isinstance(h, list):
                sqdists = []
                for emb in h:
                    emb_in = emb[i, :]
                    # emb_out = emb[idx[:, 1], :]
                    emb_out = emb[np.arange(num_users, num_users + num_items), :]
                    sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
                    sqdists.append(sqdist)
                sqdist = sum(sqdists) / len(sqdists)
            else:
                emb_in = h[i, :]
                emb_in = emb_in.repeat(num_items).view(num_items, -1)
                emb_out = h[np.arange(num_users, num_users + num_items), :]
                sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix
