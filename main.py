import argparse
import os
import pickle
from tqdm import tqdm
import pdb

import torch

from utils.data_generator import Data
from utils.helper import default_device, set_seed, argmax_top_k, ndcg_func
from utils.sampler import WarpSampler
from LGCFModel import LGCFModel
from utils.pre_utils import set_up_optimizer_scheduler
from manifolds import StiefelManifold


def train(model, data, args):
    # pass
    optimizer, lr_scheduler, stiefel_optimizer, stiefel_lr_scheduler = set_up_optimizer_scheduler(False, args, model, args.lr, args.lr_stie)

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    # print(num_batches)

    for epoch in tqdm(range(args.epoch)):
        
        for batch in tqdm(range(num_batches)):
            triples = sampler.next_batch()
            model.train()
            optimizer.zero_grad()
            stiefel_optimizer.zero_grad()
            embeddings = model.encode(data.adj_train_norm.to(args.device))
            train_loss = model.compute_loss(embeddings, triples)
            train_loss.backward()

            optimizer.step()
            stiefel_optimizer.step()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Amazon-CD', type=str)
    parser.add_argument('--c', default=1, type=float)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--tie_weight', action='store_true', default=True)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.1, help='scale for init embedding in Euclidean space')

    # optimization
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_stie', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--stiefel_optimizer', default='rsgd')
    # parser.add_argument('--weight_manifold', default="StiefelManifold")
    parser.add_argument('--lr_scheduler', default='step')

    parser.add_argument('--step_lr_gamma', default=0.1, help='gamma for StepLR scheduler')
    parser.add_argument('--step_lr_reduce_freq', default=500, help='step size for StepLR scheduler')

    args = parser.parse_args()
    print(args)
    args.device = torch.device('cuda')
    set_seed(args.seed)
    args.weight_manifold = StiefelManifold(args, 1)

    # ==== Load data ===
    processed_path = os.path.join('data', args.dataset, 'processed.pkl')
    if os.path.exists(processed_path):
        with open(processed_path, 'rb') as f:
            print(f'Loading data from {processed_path}')
            data = pickle.load(f)
    else:
        data = Data(args.dataset, norm_adj=True, seed=args.seed, test_ratio=0.2)
        with open(processed_path, 'wb') as f:
            print(f'Dumping data to {processed_path}')
            pickle.dump(data, f)

    total_edges = data.adj_train.count_nonzero()
    args.n_nodes = data.num_users + data.num_items
    # args.feat_dim = args.embedding_dim

    sampler = WarpSampler((data.num_users, data.num_items),
                          data.adj_train, args.batch_size, args.num_neg, n_workers=1)


    args.stie_vars = []
    args.eucl_vars = []

    model = LGCFModel((data.num_users, data.num_items), args).cuda()

    train(model, data, args)
    print('Finished')

