from main import lazy_load_dataset
from argparse import Namespace
import numpy as np
from scipy.stats import powerlaw
import matplotlib.pyplot as plt
import pickle
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Amazon-CD')
parser.add_argument('--n', type=int, default=2)
args = parser.parse_args()


data = lazy_load_dataset(args)
user_item = data.user_item_csr
adj = data.adj_train
item_adj = user_item.T @ user_item
user_adj = user_item @ user_item.T


for A0 in [adj, item_adj, user_adj]:
    A = A0 + 0
    assert A is not A0
    ds  = []
    ds.append(np.array((A >0).sum(1)).squeeze())
    for j in tqdm(2, range(args.n+1)):
        A = A @ A0 
        ds.append(np.array((A >0).sum(1)).squeeze())
        with open(f'{A0.__name__}_{j}_hop.pkl', 'wb') as f:
            pickle.dump(ds, f)