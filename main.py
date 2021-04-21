import argparse


from utils.data_generator import Data



def train():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Amazon-CD', type=str)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    print(args)

    data = Data(args.dataset, norm_adj=True, seed=args.seed, test_ratio=0.2)
