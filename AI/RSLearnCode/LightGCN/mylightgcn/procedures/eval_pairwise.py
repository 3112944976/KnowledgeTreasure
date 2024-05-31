import numpy as np
import torch
from tqdm import tqdm
from mylightgcn.utils import tools, measure


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()  # 预测结果转numpy数组
    ground_truth = X[1]  # 真实标签

    batch_size = sorted_items.shape[0]
    recall_scores = np.zeros(batch_size)
    precision_scores = np.zeros(batch_size)
    ndcg_scores = np.zeros(batch_size)

    for i in range(batch_size):
        true_items = set(ground_truth[i])
        pred_items = set(sorted_items[i, :topks])
        # print("true_items", true_items)
        # print("pred_items", pred_items)
        if not true_items:
            continue
        # 计算召回率
        recall_scores[i] = len(pred_items & true_items) / len(true_items)
        # 计算精确率
        precision_scores[i] = len(pred_items & true_items) / len(pred_items)
        # 计算NDCG
        relevance = np.zeros(topks)
        for j, item in enumerate(sorted_items[i, :topks]):
            if item in true_items:
                relevance[j] = 1
        ndcg_scores[i] = measure.ndcg_at_k(relevance, topks)
    # print("recall", recall_scores)
    # 计算平均分数
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(precision_scores)
    avg_ndcg = np.mean(ndcg_scores)
    return avg_recall, avg_precision, avg_ndcg


def eval_pairwise(train_dict, test_dict, model, test_batch_size, topk, device):
    model = model.eval()
    with torch.no_grad():
        # 获取所有测试用户
        users = tuple(test_dict.keys())
        # 初始化三个空列表，用于存储用户、评分和真实标签列表。
        users_list = []
        rating_list = []
        ground_truth_list = []
        # 计算总批次数
        total_batch = len(users) // test_batch_size + 1
        # 对用户进行批次处理，用进度条显示验证进度
        for batch_users in tqdm(tools.minibatch(users, batch_size=test_batch_size), desc="Validation", total=total_batch, leave=False):
            # 将当前批次用户的列表转换为 PyTorch Tensor，并移动到指定的设备上
            batch_users_gpu = torch.Tensor(batch_users).long().to(device)
            # 获取当前批次用户的真实标签:[[user1_label_items], ...]
            label = [test_dict[user] for user in batch_users]
            # 获取当前批次用户的评分
            rating = model.get_user_rating(batch_users_gpu)

            # 获取当前批次用户的所有正样本物品
            batch_users_pos = []
            for user in batch_users:
                batch_users_pos.append(train_dict[user])
            # 初始化两个空列表，用于存储要排除的索引和物品。
            exclude_index = []
            exclude_items = []
            # 对每个用户的正样本物品进行循环
            for range_i, items in enumerate(batch_users_pos):
                # 将当前用户的索引多次添加到排除索引列表中，以保持与评分向量的形状一致。
                exclude_index.extend([range_i] * len(items))
                # 将当前用户的所有正样本物品添加到排除物品列表中。
                exclude_items.extend(items)
            # 将排除物品的评分设置为一个很小的值，以便排除它们。
            rating[exclude_index, exclude_items] = -(1 << 10)
            # print("rating", rating[0].tolist())

            # 获取当前用户的前 k 个最高评分物品。
            _, rating_K = torch.topk(rating, k=topk)
            # 将当前批次的用户、评分和真实标签分别添加到对应的列表中
            users_list.append(batch_users)  # [user1, user2, ...]
            rating_list.append(rating_K.cpu())  # [[item3, item5], [item8, item3], ...]
            ground_truth_list.append(label)  # [[all_poc_item], [all_poc_item], ...]
        # 断言总批次数与用户列表的长度相等。
        assert total_batch == len(users_list)
        # print("rating_list:", np.array(rating_list[0]))
        # print("ground_truth_list", ground_truth_list)
        # 将评分列表和真实标签列表打包成一个迭代器X：[(user1_pred_K, user1_ground_items), ...]
        X = zip(rating_list, ground_truth_list)
        # print("X:", np.array(list(X)))
        # 分批次评估，并输出最终的平均结果
        precision = []
        recall = []
        ndcg = []
        for batch, x in enumerate(X):
            rec, pre, ndc = test_one_batch(x, topk)
            # print("pre", pre)
            precision.append(pre)
            recall.append(rec)
            ndcg.append(ndc)
        # print("precision:{:.4f} recall:{:.4f} ndcg:{:.4f}".format(np.mean(precision), np.mean(recall), np.mean(ndcg)))
        return np.mean(precision), np.mean(recall), np.mean(ndcg)
