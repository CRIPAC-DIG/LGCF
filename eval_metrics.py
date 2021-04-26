import numpy as np


def recall_at_k_per_user(actual, pred, k):
    act_set = set(actual)
    return len(act_set & set(pred[:k])) / float(len(act_set))


def recall_at_k(actual, predicted, user2id, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for u in user2id.keys():
        i = user2id[u]
        v = actual[u]
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)
