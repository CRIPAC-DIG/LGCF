import argparse
import os
import pickle
from tqdm import tqdm
import pdb
import sys
import numpy as np

import torch

from utils.data_generator import Data
from utils.helper import set_seed, Logger
from utils.sampler import WarpSampler
from LGCFModel import LGCFModel
from utils.pre_utils import set_up_optimizer_scheduler
from eval_metrics import recall_at_k, ndcg_func


def train(model, data, args):
    if args.eval_epoch is not None:
        # eval saved model
        model_path = os.path.join(args.log_dir, args.name + f'_model_{args.eval_epoch}.pth')
        print(f'Loading {model_path}...')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(data.adj_train_tensor.to(args.device))
            pred_matrix, user2id = model.predict(embeddings, data)
            results = eval_rec(pred_matrix, user2id, data)
        
        print(f'{args.name}\t{results[0][0]:.4f}, {results[0][1]:.4f}, {results[1][0]:.4f}, {results[1][1]:.4f}')
    else:
        optimizer, lr_scheduler = set_up_optimizer_scheduler(args, model, args.lr)

        num_pairs = data.adj_train.count_nonzero() // 2
        num_batches = int(num_pairs / args.batch_size) + 1

        for epoch in tqdm(range(args.epoch)):
            avg_loss = 0.
            for batch in tqdm(range(num_batches)):
                triples = sampler.next_batch()
                model.train()
                optimizer.zero_grad()
                embeddings = model.encode(data.adj_train_tensor.to(args.device))
                train_loss = model.compute_loss(embeddings, triples)
                train_loss.backward()

                optimizer.step()

                avg_loss += train_loss.detach().cpu().item() / num_batches
            print(f'Epoch: {epoch+1:04d} loss: {avg_loss:.4f}')

            lr_scheduler.step()
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    embeddings = model.encode(data.adj_train_tensor.to(args.device))
                    pred_matrix, user2id = model.predict(embeddings, data, args.eval_percent)
                    results = eval_rec(pred_matrix, user2id, data)
                print(
                    f'{args.name}\t{results[0][0]:.4f}, {results[0][1]:.4f}, {results[1][0]:.4f}, {results[1][1]:.4f}')
                torch.save(model.state_dict(), os.path.join(args.log_dir, args.name + f'_model_{epoch+1}.pth'))

def eval_rec(pred_matrix, user2id, data):
    """

    user2id: user to row index of pred_matrix
    """
    topk = 20
    mask = np.array([True  if u in user2id else False for u in data.user_item_csr.nonzero()[0]])
    users, items = data.user_item_csr.nonzero()
    users = np.array(users)
    items = np.array(items)
    users = [user2id[u] for u in users[mask]]
    items = items[mask]
    pred_matrix[users, items] = np.NINF

    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    recall = []
    Ks = (10, 20)
    for k in Ks:
        recall.append(recall_at_k(data.test_dict, pred_list, user2id, k))

    gt_list = [data.test_dict[u] for u in user2id]
    all_ndcg = ndcg_func(gt_list, pred_list)
    ndcg = [all_ndcg[x-1] for x in Ks]

    return recall, ndcg

def lazy_load_dataset(args):
    processed_path = os.path.join('data', args.dataset, 'processed.pkl')
    if os.path.exists(processed_path):
        with open(processed_path, 'rb') as f:
            print(f'Loading data from {processed_path}')
            data = pickle.load(f)
    else:
        data = Data(args.dataset, seed=args.seed, test_ratio=0.2)
        with open(processed_path, 'wb') as f:
            print(f'Dumping data to {processed_path}')
            pickle.dump(data, f)
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Amazon-CD', type=str)
    parser.add_argument('--c', default=1, type=float)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.1, help='scale for init embedding in Euclidean space')

    # optimization
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr_scheduler', default='step')
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--step_lr_gamma', default=0.1, help='gamma for StepLR scheduler')
    parser.add_argument('--step_lr_reduce_freq', default=30, help='step size for StepLR scheduler')

    parser.add_argument('--res_sum', action='store_true', default=False)

    parser.add_argument('--eval_percent', type=int, default=100)
    parser.add_argument('--eval_epoch', type=int, default=None)

    args = parser.parse_args()

    log_dir = f'log/{args.dataset}/margin_loss_no_weight'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir
    args.name = f'eval_{args.eval_percent}_layer_{args.num_layers}_dim_{args.embedding_dim}_neg_{args.num_neg}_res_{args.res_sum}_batch_{args.batch_size}_lr_{args.lr}_{args.step_lr_reduce_freq}_{args.step_lr_gamma}_decay_{args.weight_decay}_margin_{args.margin}'
    log_file = args.name + '_log.txt'
    log_file_path = os.path.join(log_dir, log_file)
    sys.stdout = Logger(log_file_path)

    print(args)
    args.device = torch.device('cuda')
    set_seed(args.seed)
    
    data = lazy_load_dataset(args)
    args.n_nodes = data.num_users + data.num_items
    sampler = WarpSampler((data.num_users, data.num_items),
                          data.adj_train, args.batch_size, args.num_neg, n_workers=1)

    args.eucl_vars = []
    model = LGCFModel((data.num_users, data.num_items), args).cuda()

    train(model, data, args)
    print('Finished')
    sampler.close()
