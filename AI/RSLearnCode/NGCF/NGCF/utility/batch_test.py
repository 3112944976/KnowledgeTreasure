'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
# from load_data import *
# from parser import parse_args
# import metrics

import multiprocessing
import heapq
import numpy as np
# 确定并行处理任务时使用的CPU核心数
cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)  # 将字符串形式的列表转换为实际的列表对象
# 1. 加载并处理数据
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
# 获得用户数和物品数
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
# 获得训练总交互数和测试总交互数
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


# def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
#     """
#     功能：生成推荐列表并计算AUC
#     参数：
#         user_pos_test: 用户在测试集上的正样本集合
#         test_items：待评估的物品集合
#         rating：物品的评分字典
#     """
#     # 1. 遍历用户u的每个训练未观测物品i，获得对应预测评分，存储到item_score中："i":score。
#     item_score = {}
#     for i in test_items:
#         item_score[i] = rating[i]
#
#     K_max = max(Ks)  # K_max=100
#     # 2. 获取评分最高的前K_max个物品
#     K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
#     # 3. 计算该用户的AUC：遍历推荐列表中的每个物品，若为正样本则为1，否则为0
#     r = []
#     for i in K_max_item_score:
#         if i in user_pos_test:
#             r.append(1)
#         else:
#             r.append(0)
#     auc = 0.
#     return r, auc


def get_auc(item_score, user_pos_test):
    """
    功能：计算给定推荐列表的AUC值
    Args:
        item_score: 用户u对所有训练未观测物品的预测评分字典
        user_pos_test: test_set[u]用户u在测试集中所有的标签物品
    """
    # 将item_score按照评分排序
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]  # 排序后的物品id列表
    posterior = [x[1] for x in item_score]  # 排序后的分数列表
    # 获取对应的真实标签r
    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    # 计算AUC
    # r：用户u对所有未观测物品的命中标记（已根据预测值排序过）
    # posterior：对应r中每个标记的预测评分（已根据预测值排序过）
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    """基于排序生成推荐列表，并计算 AUC"""
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    """计算prec recall ndcg hit"""
    precision, recall, ndcg, hit_ratio = [], [], [], []
    # 分别计算不同TOP-N下的度量值
    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    """输入参数x:
        work：某批次中所有用户对全部物品的预测评分矩阵
        size：user_batch*ITEM_NUM
    """
    # 1. 用户u对所有物品的评分
    rating = x[0]
    # 2. 用户u的id
    u = x[1]
    # 3. 获取用户u的训练观测物品集
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    # 4. 获取用户u的测试物品集
    user_pos_test = data_generator.test_set[u]
    # 创建一个包含所有物品索引的集合
    all_items = set(range(ITEM_NUM))
    # 5. 获取用户u的训练未观测物品集
    test_items = list(all_items - set(training_items))
    # # 6. 若进行批量测试（二选一）
    # if args.test_flag == 'part':
    #     r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    # # 6. 若进行全物品测试（二选一）
    # else:
    #     r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    # 将用户u在测试集上的真实物品列表user_pos_test，标签列表r，计算而来的auc和top-K值列表Ks。
    return get_performance(user_pos_test, r, auc, Ks)


def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    """
    Args:
        model: NGCF模型
        users_to_test: 测试用户列表
        drop_flag: 布尔类型，控制是否使用dropout
        batch_test_flag: 布尔类型，控制是否进行批量测试
    Returns: 性能指标字典result
    """
    # 1. 初始化结果字典
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    # 2. 创建资源池，使用cores个进程并行处理任务
    pool = multiprocessing.Pool(cores)

    # 设置批次用户数和物品数
    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE
    # 获取测试用户列表和数量
    test_users = users_to_test
    n_test_users = len(test_users)
    # 若每次取u_batch_size个用户数，计算需要n_user_batchs个批次才能完成测试。
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0
    # 遍历每个用户批次
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        # 3. 获取当前批次的用户集
        user_batch = test_users[start: end]
        # 4. 所有物品分批测试（二选一）
        if batch_test_flag:
            # 计算物品批次数目
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            # 初始化评分批次矩阵，shape=(当前批次的用户数, 物品总数)
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))
            # 遍历每个物品批次
            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)
                # 构建当前物品批次的索引范围
                item_batch = range(i_start, i_end)
                # 获取模型评分，不使用dropout
                if not drop_flag:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [], drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                # 获取模型评分，使用dropout
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [], drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                # 将当前物品批次的评分放入评分批次矩阵中
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]
            # 确保处理的评分数目等于总物品数目
            assert i_count == ITEM_NUM
        # 4. 所有物品一起测试（二选一）
        else:
            item_batch = range(ITEM_NUM)
            if not drop_flag:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [], drop_flag=False)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch, item_batch, [], drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
        # 5. 将批次用户评分矩阵和批次用户集组成为元组：user_batch_rating_uid=(user_id, len([0.5, 0.7, 0.3, ...])=ITEM_NUM)
        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        # 6. 使用pool.map并行调用函数test_one_user，每次仅处理一个用户及其对应长度为ITEM_NUM的评分结果
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)  # 累计处理的用户数目
        # 遍历当前批次用户集合下，其中每个用户u的评估结果re。n_test_users为当前批次的用户总数。
        for re in batch_result:
            # 计算当前批次所有测试用户，评价结果的均值。
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
    # 确保处理的用户数目等于测试用户总数
    assert count == n_test_users
    pool.close()
    return result
