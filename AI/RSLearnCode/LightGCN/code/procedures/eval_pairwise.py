import multiprocessing
import utils
import world
import numpy as np
import torch

from tqdm import tqdm


# 评估模型在一个测试数据批次上的性能
def test_one_batch(X, item_embeddings, batch_user_bins, batch_user_interaction_history, num_bins=20):
    """
    Args:
        X (tuple): 一批包含(预测值,标签)的测试数
        item_embeddings (torch.Tensor): 物品嵌入向量
        batch_user_bins (list): 用户分组列表，指示当前批次中每个用户根据交互次数所对应的分组
        batch_user_interaction_history (list): 用户交互历史记录列表，包含了当前批次中每个用户的交互历史记录
        num_bins (int, optional): 用于对用户交互进行分组的分组数量。默认为 20。
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # 从测试数据中提取出预测评分，并转换为numpy数组
    sorted_items = X[0].numpy()
    # 从测试数据中提取出真实标签
    ground_truth = X[1]
    # 获得每个用户对物品的评分情况
    label = utils.get_label(ground_truth, sorted_items)
    # 初始化三个空列表，用于存储不同评价指标的结果
    precision, recall, ndcg = [], [], []
    # 初始化一个二维数组，用于记录探索与精确度间的关系
    exploration_vs_precision = np.zeros((len(world.topks), num_bins))
    # 初始化一个二维数组，用于记录探索与召回率间的关系
    exploration_vs_recall = np.zeros((len(world.topks), num_bins))
    # 初始化一个二维数组，用于记录探索与NDCG间的关系
    exploration_vs_ndcg = np.zeros((len(world.topks), num_bins))
    # 对每个指定的K值进行循环，world.topks是一个包含了要计算的不同k值的列表
    for i_k, k in enumerate(world.topks):
        # 计算在给定k值下的召回率和精确率
        ret = utils.recall_precision_at_k(ground_truth, label, k)
        precision.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.ndcg_at_k_r(ground_truth, label, k))
        # 对每个用户进行循环迭代
        for user in range(len(sorted_items)):
            # 获取当前用户所属的交互次数分组
            user_bin = batch_user_bins[user]
            # 将当前用户的真实标签转换为一个二维数组
            user_ground_truth = np.expand_dims(ground_truth[user], axis=0)
            # 将当前用户的标签转换为一个二维数组
            user_label = np.expand_dims(label[user, :], axis=0)
            # 计算当前用户在给定 k 值下的召回率和精确度
            user_ret = utils.recall_precision_at_k(user_ground_truth, user_label, k)
            user_precision = user_ret["precision"]
            user_recall = user_ret["recall"]
            # 计算当前用户在给定 k 值下的NDCG
            user_ndcg = utils.ndcg_at_k_r(user_ground_truth, user_label, k)
            # 更新探索与精确度或召回率之间的关系
            exploration_vs_precision[i_k, user_bin] += user_precision
            exploration_vs_recall[i_k, user_bin] += user_recall
            exploration_vs_ndcg[i_k, user_bin] = user_ndcg
    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "diversity": utils.mean_intra_list_distance(
            recommendation_lists=sorted_items,
            item_embeddings=item_embeddings
        ),
        "novelty": utils.novelty(
            ground_truth, batch_user_interaction_history,
            max(world.topks)
        ),
        "exploration_vs_precision": exploration_vs_precision,
        "exploration_vs_recall": exploration_vs_recall,
        "exploration_vs_ndcg": exploration_vs_ndcg
    }


# Evaluate the pairwise ranking model on the test data.
def eval_pairwise(dataset, model, multicore=0):
    """
    Args:
        dataset (BasicDataset): The dataset containing user-item interactions.
        model (BasicModel): The pairwise ranking model.
        multicore (int, optional): Number of cores to use for parallel processing. Defaults to 0.
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    batch_size = world.config["test_u_batch_size"]
    test_dict = dataset.test_dict

    model = model.eval()
    # 获得要考虑的最大k值
    max_k = max(world.topks)
    # 如若启用了多核计算
    if multicore:
        # 设置多进程模式为 spawn 模式，用于多进程处理
        multiprocessing.set_start_method("spawn", force=True)
        # 创建一个进程池对象
        pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)

    # 获取用户交互次数分组的数量
    num_bins = world.num_bins
    # 初始化一个字典，用于存储评价指标结果
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "diversity": 0.,
        "novelty": 0.,
        "exploration_vs_precision": np.zeros((len(world.topks), num_bins)),
        "exploration_vs_recall": np.zeros((len(world.topks), num_bins)),
        "exploration_vs_ndcg": np.zeros((len(world.topks), num_bins))
    }
    # 确保在此期间不会计算梯度
    with torch.no_grad():
        # 获取所有用户的列表
        users = list(test_dict.keys())
        # 初始化三个空列表，用于存储用户、评分和真实标签列表。
        users_list = []
        rating_list = []
        ground_truth_list = []
        # 计算总批次数
        total_batch = len(users) // batch_size + 1
        # 对用户进行批次处理，用进度条显示验证进度
        for batch_users in tqdm(utils.minibatch(users, batch_size=batch_size), desc="Validation", total=total_batch, leave=False):
            # 将当前批次用户的列表转换为 PyTorch Tensor，并移动到指定的设备上
            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)

            # 获取当前批次用户的真实标签
            ground_truth = [test_dict[user] for user in batch_users]
            # 获取当前批次用户的评分
            rating = model.get_user_rating(batch_users_gpu)
            # 获取当前批次用户的所有正样本物品
            all_pos = dataset.get_user_pos_items(batch_users)

            # 初始化两个空列表，用于存储要排除的索引和物品。
            exclude_index = []
            exclude_items = []

            # 对每个用户的正样本物品进行循环
            for range_i, items in enumerate(all_pos):
                # 将当前用户的索引多次添加到排除索引列表中，以保持与评分向量的形状一致。
                exclude_index.extend([range_i] * len(items))
                # 将当前用户的所有正样本物品添加到排除物品列表中。
                exclude_items.extend(items)
            # 将排除物品的评分设置为一个很小的值，以便排除它们。
            rating[exclude_index, exclude_items] = -(1 << 10)
            # 获取当前用户的前 k 个最高评分物品。
            _, rating_K = torch.topk(rating, k=max_k)
            # 将评分张量转换为 numpy 数组。
            rating = rating.cpu().numpy()

            del rating
            # 将当前批次的用户、评分和真实标签分别添加到对应的列表中
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            ground_truth_list.append(ground_truth)
        # 断言总批次数与用户列表的长度相等。
        assert total_batch == len(users_list)

        # 将评分列表和真实标签列表打包成一个迭代器X。
        X = zip(rating_list, ground_truth_list)
        # 获取模型的物品嵌入向量
        _, item_embeddings = model()
        # 计算用户根据交互次数分组的列表
        user_bins_by_num_interactions = [
            [dataset.user_bins_by_num_interactions[user_id]
             for user_id in batch_users] for batch_users in users_list
        ]
        # 计算用户的交互历史记录
        user_interaction_history = [
            [dataset.user_interactions_dict_train[user_id]
             for user_id in batch_users] for batch_users in users_list
        ]

        # 如果启用了多核计算
        if multicore:
            # 使用多进程并行计算每个批次的评价指标
            pre_results = pool.starmap(
                test_one_batch,
                [(x, item_embeddings, user_bins_by_num_interactions[batch],
                 user_interaction_history[batch], num_bins)
                 for batch, x in enumerate(X)]
            )
        else:
            # 否则，逐批次计算评价指标
            pre_results = []
            for batch, x in enumerate(X):
                pre_results.append(
                    test_one_batch(
                        x, item_embeddings,
                        user_bins_by_num_interactions[batch],
                        user_interaction_history[batch], num_bins
                    )
                )
        # 对每个批次的评价指标结果进行汇总
        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["diversity"] += result["diversity"]
            results["novelty"] += result["novelty"]
            results["exploration_vs_precision"] += \
                result["exploration_vs_precision"]
            results["exploration_vs_recall"] += result["exploration_vs_recall"]
    return results