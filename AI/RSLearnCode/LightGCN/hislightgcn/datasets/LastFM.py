from .BasicDataset import BasicDataset
from os.path import join
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
import torch
import utils
import world


class LastFM(BasicDataset):
    def __init__(self, path=join("..", "data", "lastfm")):
        print("loading [last fm]")
        # 初始化一个字典，表示训练和测试模式
        self.mode_dict = {"train": 0, "test": 1}
        # 将模式设置为训练模式
        self.mode = self.mode_dict["train"]
        # 设置用户和物品数
        self.n_user = 1892
        self.m_item = 4489
        # 读取训练和测试数据文件
        train_data = pd.read_table(join(path, "data1.txt"), header=None)
        test_data = pd.read_table(join(path, "test1.txt"), header=None)
        # 读取信任网络数据文件，并转换为 NumPy 数组
        trust_net = pd.read_table(join(path, "trustnetwork.txt"), header=None).to_numpy()
        # 将所有数据减去1，以匹配Python中的0-based索引
        trust_net -= 1
        train_data -= 1
        test_data -= 1
        # 将数据存储在类的属性中
        self.trust_net = trust_net
        self.train_data = train_data
        self.test_data = test_data
        # 提取训练和测试数据的用户和物品信息，并找到唯一的用户
        self.train_user = np.array(train_data[:][0])
        self.train_unique_users = np.unique(self.train_user)
        self.train_item = np.array(train_data[:][1])
        self.test_user = np.array(test_data[:][0])
        self.test_unique_user = np.unique(self.test_user)
        self.test_item = np.array(test_data[:][1])
        # 初始化图属性
        self.graph = None
        # 计算数据稀疏度，并打印结果
        dataset_sparsity = len(self.train_user) + len(self.test_user)
        dataset_sparsity /= self.n_user
        dataset_sparsity /= self.m_item
        print(f"LastFm Sparsity : {dataset_sparsity}")
        # 构建稀疏矩阵，表示社交网络和用户-物品网络
        self.social_net = csr_matrix(
            (np.ones(len(trust_net)), (trust_net[:, 0], trust_net[:, 1])),
            shape=(self.n_users, self.n_users))
        self.user_item_net = csr_matrix(
            (
                np.ones(len(self.train_user)),
                (self.train_user, self.train_item)
            ),
            shape=(self.n_users, self.m_items))

        # pre-calculate
        self.all_positive = self.get_user_pos_items(list(range(self.n_users)))
        self.all_negative = []
        all_items = set(range(self.m_items))

        for i in range(self.n_users):
            pos = set(self.all_positive[i])
            neg = all_items - pos
            self.all_negative.append(np.array(list(neg)))
        # 构建测试数据字典
        self.__testDict = self.__build_test()
    # @property装饰器，用于定义属性访问器
    # 返回用户数量
    @property
    def n_users(self):
        return self.n_user
    # 返回物品数量
    @property
    def m_items(self):
        return self.m_item
    # 返回训练数据的大小
    @property
    def train_data_size(self):
        return len(self.train_user)
    # 返回测试数据字典
    @property
    def test_dict(self):
        return self.__testDict
    # 返回所有正样本
    @property
    def all_pos(self):
        return self.all_positive

    def get_sparse_graph(self):
        # 如果图为空，则构建一个稀疏张量来表示图
        if self.graph is None:
            user_dim = torch.LongTensor(self.train_user)
            item_dim = torch.LongTensor(self.train_item)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])

            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.graph = torch.sparse.IntTensor(
                index, data,
                torch.Size([self.n_users + self.m_items,
                            self.n_users + self.m_items])
            )

            dense = self.graph.to_dense()

            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()

            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)

            self.graph = torch.sparse.FloatTensor(
                index.t(), data,
                torch.Size([self.n_users+self.m_items,
                            self.n_users+self.m_items])
            )

            self.graph = self.graph.coalesce().to(world.device)

        return self.graph
    # 构建一个字典，将用户映射到其测试物品列表
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]

            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data
    # 获取给定用户和物品的反馈
    def get_user_item_feedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        user_item_feedback = np.array(self.user_item_net[users, items])
        user_item_feedback = user_item_feedback.astype("uint8")
        user_item_feedback = user_item_feedback.reshape((-1, ))

        return user_item_feedback
    # 获取给定用户的正样本物品
    def get_user_pos_items(self, users):
        pos_items = []

        for user in users:
            pos_items.append(self.user_item_net[user].nonzero()[1])

        return pos_items
    # 定义了当对该类的实例进行索引时的行为，返回训练集中的唯一用户
    def __getitem__(self, index):
        user = self.train_unique_users[index]

        return user
    # 将数据集模式切换为测试模式
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict["test"]
    # 返回训练集中唯一用户的数量
    def __len__(self):
        return len(self.train_unique_users)
