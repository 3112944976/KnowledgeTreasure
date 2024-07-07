'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.n_users, self.n_items = 0, 0  # 初始化用户和物品数量
        self.n_train, self.n_test = 0, 0  # 初始化训练和测试交互数量
        self.neg_pools = {}  # 存储负样本池
        self.exist_users = []  # 存储存在的用户列表

        # 1. 读取训练文件，更新用户数n_users和物品数n_items、获取训练交互数n_train
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))  # 更新最大物品索引
                    self.n_users = max(self.n_users, uid)  # 更新最大用户索引
                    self.n_train += len(items)  # 增加训练交互数量
        # 2. 读取测试文件，更新物品数n_items、获取测试交互数n_test
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))  # 更新最大物品索引
                    self.n_test += len(items)  # 增加测试交互数量
        # 物品和用户的索引从0开始，因此数量需要加1
        self.n_items += 1
        self.n_users += 1
        # # 3. 打印数据统计信息
        # self.print_statistics()
        # 4. 获得稀疏用户-物品交互矩阵R、train_items字典、test_items字典
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # 初始化稀疏矩阵R
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                # 遍历训练集中的行数据，填充R并往train_items中追加对应的："uid": item[1:]
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    for i in train_items:
                        self.R[uid, i] = 1.
                    self.train_items[uid] = train_items
                # 遍历测试集中的行数据，往test_items中追加对应的："uid": item[1:]
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
    # # 加载邻接矩阵和规范化的邻接矩阵
    # def get_adj_mat(self):
    #     try:
    #         t1 = time()
    #         adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
    #         norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
    #         mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
    #         print('已加载预保存的稀疏邻接矩阵adj_mat', adj_mat.shape, time() - t1)
    #     except Exception:
    #         # 创建稀疏邻接矩阵adj_mat、规范化稀疏矩阵norm_adj_mat、均值稀疏矩阵mean_adj_mat
    #         adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
    #         sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
    #         sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
    #         sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
    #     return adj_mat, norm_adj_mat, mean_adj_mat
    # 加载规范邻接矩阵
    def get_adj_mat(self):
        try:
            t1 = time()
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            print('已加载预保存的稀疏norm_adj_mat', norm_adj_mat.shape)
        except Exception:
            norm_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
        return norm_adj_mat
    # 创建规范邻接矩阵
    def create_adj_mat(self):
        t1 = time()
        # 初始化稀疏邻接矩阵adj_mat，维度为(n_users+n_items, n_items+n_users)
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        # 稀疏adj_mat和R稠密化
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        # 覆写数据
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        # adj_mat稀疏化
        adj_mat = adj_mat.todok()
        print('已创建稀疏adj_mat，归一化处理中...', adj_mat.shape)

        t2 = time()
        # 均值归一化函数
        def mean_adj_single(adj):
            # D^(-1)*A
            rowsum = np.array(adj.sum(1))  # 计算每行的和
            d_inv = np.power(rowsum, -1).flatten()  # 计算每行和的倒数，并展开为一维数组
            d_inv[np.isinf(d_inv)] = 0.  # 将无穷大的值置为0
            d_mat_inv = sp.diags(d_inv)  # 创建对角稀疏矩阵
            norm_adj = d_mat_inv.dot(adj)  # 计算规范化邻接矩阵
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
        # 双向拉普拉斯归一化函数
        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        # def check_adj_if_equal(adj):
        #     dense_A = np.array(adj.todense())
        #     degree = np.sum(dense_A, axis=1, keepdims=False)
        #
        #     temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        #     print('check normalized adjacency matrix whether equal to this laplacian matrix.')
        #     return temp

        # 邻接矩阵归一化处理
        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))  # D^(-1)*(A+I)
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))  # D^(-1/2)*(A+I)*D^(1/2)
        # norm_adj_mat = normalized_adj_single(adj_mat) + sp.eye(adj_mat.shape[0]).tocoo()  # D^(-1/2)*A*D^(1/2)+I
        # mean_adj_mat = mean_adj_single(adj_mat)  # D^(-1)*(A)
        print('已生成稀疏norm_adj_mat')
        # return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
        return norm_adj_mat.tocsr()
    # # 生成负样本池
    # def negative_pool(self):
    #     t1 = time()
    #     # 遍历训练集中的所有用户
    #     for u in self.train_items.keys():
    #         # 获得用户u的所有负样本
    #         neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
    #         # 随机采样100个负样本
    #         pools = [rd.choice(neg_items) for _ in range(100)]
    #         # 追加存储到负样本池中，即："u":pools
    #         self.neg_pools[u] = pools
    #     print('已刷新负样本池', time() - t1)
    # 采样用户和对应的正负样本
    def sample(self):
        # 随机选择batch_size个用户
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]  # 获取用户的训练物品列表
            n_pos_items = len(pos_items)  # 计算训练物品列表的长度
            pos_batch = []
            while True:
                if len(pos_batch) == num:  # 若正样本批次达到指定数量，则结束。
                    break
                # 在指定范围内随机生成一个整数作为索引
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                # 获取对应索引的物品id
                pos_i_id = pos_items[pos_id]
                # 如果物品id不在正样本批次中，则添加到pos_batch中
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:  # 若负样本批次达到指定数量，则结束。
                    break
                # 在指定范围内随机生成一个整数作为索引
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                # 如果负样本不在用户的训练集和当前负样本列表中，则存储
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        # def sample_neg_items_for_u_from_pools(u, num):
        #     neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
        #     return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:  # 遍历所有用户
            pos_items += sample_pos_items_for_u(u, 1)  # 采样每个用户的一个正样本
            neg_items += sample_neg_items_for_u(u, 1)  # 采样每个用户的一个负样本

        return users, pos_items, neg_items
    # # 获取用户和物品的数量
    # def get_num_users_items(self):
    #     return self.n_users, self.n_items
    # # 打印数据统计信息
    # def print_statistics(self):
    #     print('总用户数n_users=%d, 总物品数n_items=%d' % (self.n_users, self.n_items))
    #     print('总交互数n_interactions=%d' % (self.n_train + self.n_test))
    #     print('训练交互数n_train=%d, 测试交互数n_test=%d, 数据稀疏度sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    # # 获取稀疏度划分信息
    # def get_sparsity_split(self):
    #     try:
    #         # 初始化两个空列表，存储稀疏度划分信息中的用户id和状态信息
    #         split_uids, split_state = [], []
    #         # 打开稀疏度划分信息文件，并逐行读取内容
    #         lines = open(self.path + '/sparsity.split', 'r').readlines()
    #         # 1. 遍历文件中的行索引及行内容
    #         for idx, line in enumerate(lines):
    #             if idx % 2 == 0:  # 若当前行的索引为偶数
    #                 split_state.append(line.strip())  # 去除line中的换行符并添加到split_state列表中
    #                 print(line.strip())  # 打印去除换行符后的line内容
    #             else:  # 若当前行的索引为奇数
    #                 # 将去除换行符后按空格分割的字符串转换为整数列表，并添加到split_uids列表中
    #                 split_uids.append([int(uid) for uid in line.strip().split(' ')])
    #         print('已获取稀疏度划分信息')
    #
    #     except Exception:
    #         split_uids, split_state = self.create_sparsity_split()  # 创建稀疏度划分信息
    #         f = open(self.path + '/sparsity.split', 'w')  # 打开稀疏度划分信息文件以便写入
    #         for idx in range(len(split_state)):  # 遍历split_state列表中每个元素的索引idx
    #             # 将每个稀疏度划分状态写入文件并换行
    #             f.write(split_state[idx] + '\n')
    #             # 将每个稀疏度划分uid列表写入文件并换行
    #             f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
    #         print('已创建稀疏度划分信息')
    #     # 返回稀疏度划分uid列表和稀疏度划分状态列表
    #     return split_uids, split_state
    # # 创建稀疏度划分信息
    # def create_sparsity_split(self):
    #     all_users_to_test = list(self.test_set.keys())  # 获取测试用户id列表
    #     user_n_iid = dict()  # 初始化用户n_iid字典，存储：“物品总数":[uid,...]
    #     # 遍历所有测试用户id
    #     for uid in all_users_to_test:
    #         train_iids = self.train_items[uid]  # 获取用户uid的训练物品列表
    #         test_iids = self.test_set[uid]  # 获取用户uid的测试物品列表
    #         n_iids = len(train_iids) + len(test_iids)  # 计算用户uid的物品总数
    #         # 当第一次遇到该物品总数时
    #         if n_iids not in user_n_iid.keys():
    #             user_n_iid[n_iids] = [uid]  # 添加键值对（物品总数，用户id列表）
    #         else:
    #             # 否则，将用户id添加到user_n_iid中对应总交互数Key所在的uid列表中
    #             user_n_iid[n_iids].append(uid)
    #     # 初始化split_uids，用于存储分割后的用户id列表
    #     split_uids = list()
    #
    #     # 将整个用户集合分割成四个子集
    #     temp = []  # 用于临时存储分组的用户id
    #     count = 1
    #     fold = 4
    #     n_count = (self.n_train + self.n_test)  # 初始化为训练集和测试集物品总数之和
    #     n_rates = 0
    #
    #     split_state = []  # 用于存储每个分组的状态信息
    #     # 遍历排序后的user_n_iid字典的每个物品总数n_iids及其索引idx
    #     for idx, n_iids in enumerate(sorted(user_n_iid)):
    #         temp += user_n_iid[n_iids]  # 累计记忆当前分组的所有用户id
    #         n_rates += n_iids * len(user_n_iid[n_iids])  # 计算当前分组中所有用户的评分总数
    #         n_count -= n_iids * len(user_n_iid[n_iids])  # 减去当前分组中所有用户的物品总数
    #
    #         if n_rates >= count * 0.25 * (self.n_train + self.n_test):
    #             # 将当前分组的所有用户id添加到split_uids列表中，表示完成一个分组
    #             split_uids.append(temp)
    #             # 生成当前分组的状态信息: 物品总数n_iids 用户数量len(temp) 总评分数n_rates
    #             state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
    #             split_state.append(state)
    #             print(state)
    #             # 清空temp和n_rates，减少fold计数器的值，表示还剩多少个分组未完成。
    #             temp = []
    #             n_rates = 0
    #             fold -= 1
    #         # 若遍历到最后一个物品总数键，或者所有物品总数都已分组完毕
    #         if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
    #             split_uids.append(temp)
    #             state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
    #             split_state.append(state)
    #             print(state)
    #
    #     return split_uids, split_state
