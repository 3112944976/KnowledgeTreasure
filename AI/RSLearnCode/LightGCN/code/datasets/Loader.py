from .BasicDataset import BasicDataset
import world
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import torch
from time import time
import scipy.sparse as sp
from os.path import join


class Loader(BasicDataset):
    def __init__(self, config, path):
        super().__init__()
        print(f"loading [{path}]")  # 打印加载路径

        self.split = config["A_split"]
        self.folds = config["adj_matrix_folds"]
        self.mode_dict = {"train": 0, "test": 1}  # 定义了一个模式字典，将字符串模式映射到整数
        self.mode = self.mode_dict["train"]  # 将模式设置为训练模式
        # 初始化用户和项目数为0
        self.n_user = 0
        self.m_item = 0
        # 构建构建训练和测试文件路径
        train_file = join(path, "train.txt")
        test_file = join(path, "test.txt")
        self.path = path  # 将路径存储在实例变量中
        # 初始化训练和测试数据的空列表、数据大小和配置
        train_unique_users, train_item, train_user = [], [], []
        test_unique_users, test_item, test_user = [], [], []
        self.train_data_size = 0
        self.test_data_size = 0
        self.config = config
        # 建个默认字典，存储训练数据的用户与物品交互情况 {user_id -> [item_ids]}
        self.user_interactions_dict_train = defaultdict(list)
        # 打开训练文件，循环读取每一行数据并处理
        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")
                    # 解析每行数据中的物品列表
                    items = [int(i) for i in line[1:]]
                    # 解析每行数据中的用户ID
                    uid = int(line[0])
                    # 将用户ID、物品列表等信息存储到相应的列表中
                    train_unique_users.append(uid)
                    train_user.extend([uid] * len(items))
                    train_item.extend(items)

                    # Maintain a mapping of user_id -> [item_ids]
                    self.user_interactions_dict_train[uid].extend(items)
                    # 更新最大用户ID和物品ID
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    # 通过读取的数据更新训练数据的大小等信息
                    self.train_data_size += len(items)
        # 对训练数据进行numpy数组化
        self.train_unique_users = np.array(train_unique_users)
        self.train_user = np.array(train_user)
        self.train_item = np.array(train_item)
        # 打开测试文件，同上，循环读取每一行数据并处理
        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")

                    try:
                        items = [int(i) for i in line[1:]]
                    except Exception:
                        continue

                    uid = int(line[0])

                    test_unique_users.append(uid)
                    test_user.extend([uid] * len(items))
                    test_item.extend(items)
                    
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)

                    self.test_data_size += len(items)
        # 将最大物品ID和用户ID增加1，以确保索引从1开始
        self.m_item += 1
        self.n_user += 1
        # 对测试数据进行numpy数组化
        self.test_unique_users = np.array(test_unique_users)
        self.test_user = np.array(test_user)
        self.test_item = np.array(test_item)

        self.Graph = None
        # 打印交互数据的大小信息
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.test_data_size} interactions for testing")
        # 计算数据稀疏度
        dataset_sparsity = self.train_data_size + self.test_data_size
        # 将稀疏度归一化
        dataset_sparsity /= self.n_users
        dataset_sparsity /= self.m_items
        # 打印数据集的稀疏度
        print(f"{world.dataset} Sparsity : {dataset_sparsity}")
        # 构建用户-物品稀疏矩阵，并进行归一化处理
        self.user_item_net = csr_matrix(
            (
                np.ones(len(self.train_user)),
                (self.train_user, self.train_item)
            ),
            shape=(self.n_user, self.m_item)
        )
        # 分别计算用户和物品的度
        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # 构建所有用户的正例项目列表，并构建测试字典
        self._allPos = self.get_user_pos_items(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        # 将用户根据其交互数分配到不同的区间中
        self.user_bins_by_num_interactions = self.distribute_users_into_bins_by_num_interactions(num_bins=world.num_bins)
        # 打印数据集准备就绪信息
        print(f"{world.dataset} is ready to go")
    # 定义了一个计算用户数量的属性
    @property
    def n_users(self):
        return self.n_user
    # 定义了一个计算物品数量的属性
    @property
    def m_items(self):
        return self.m_item
    # 定义了一个获取测试字典的属性
    @property
    def test_dict(self):
        return self.__testDict
    # 定义了一个获取所有正例项目列表的属性
    @property
    def all_pos(self):
        return self._allPos
    # 分割邻接矩阵
    def __split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds

        for i_fold in range(self.folds):
            start = i_fold*fold_len

            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold.append(
                self.__convert_sp_mat_to_sp_tensor(
                    A[start: end]).coalesce().to(world.device)
            )

        return A_fold
    # 将稀疏矩阵转换为稀疏张量
    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)

        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()

        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    # 获取稀疏图
    def get_sparse_graph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            # try:
            #     pre_adj_mat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
            #     print("successfully loaded...")
            #     norm_adj = pre_adj_mat
            # except Exception:
            print("generating adjacency matrix")
            # start_time = time()

            num_nodes = self.n_users + self.m_items
            adj_mat = sp.dok_matrix(
                (num_nodes, num_nodes), dtype=np.float32)
            adj_mat = adj_mat.tolil()

            R = self.user_item_net.tolil()
            adj_mat[: self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, : self.n_users] = R.T
            adj_mat = adj_mat.todok()

            if not self.config["l1"] and (self.config["side_norm"].lower() == "l" or self.config["side_norm"].lower() == "r"):
                rowsum = np.array(np.square(adj_mat).sum(axis=1))
                # rowsum = np.square(np.array(adj_mat.sum(axis=1))
                # rowsum_squared = np.square(adj_mat).sum(axis=1)
                # rowsum = np.sqrt(rowsum_squared)
            else:
                rowsum = np.array(adj_mat.sum(axis=1))

            # L1 normalization
            exponent = -1 if self.config["l1"] else -0.5
            d_inv = np.power(rowsum, exponent).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            # left normalization
            if self.config["side_norm"].lower() == "l":
                norm_adj = d_mat.dot(adj_mat)

            # right normalization
            elif self.config["side_norm"].lower() == "r":
                norm_adj = adj_mat.dot(d_mat)

            # symmetric normalization
            elif self.config["side_norm"].lower() == "both":
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)

            norm_adj = norm_adj.tocsr()

            # end_time = time()
            # print(f"costing {end_time-start_time}s, saved norm_mat...")
            # sp.save_npz(join(self.path, "s_pre_adj_mat.npz"), norm_adj)

            if self.split:
                self.Graph = self.__split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")

        return self.Graph
    # 构建测试数据字典
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
    # 获取用户-物品反馈
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
        user_item_feedback = user_item_feedback.reshape((-1,))

        return user_item_feedback
    # 获取用户正例项目
    def get_user_pos_items(self, users):
        posItems = []

        for user in users:
            posItems.append(self.user_item_net[user].nonzero()[1])

        return posItems
    # 根据交互数将用户分配到不同区间
    def distribute_users_into_bins_by_num_interactions(self, num_bins):
        log_values = [np.log(len(self.user_interactions_dict_train[user])) for user in self.user_interactions_dict_train.keys()]

        # Create bins
        min_num_interactions = min(log_values)
        max_num_interactions = max(log_values)
        bin_thresholds = np.linspace(min_num_interactions, max_num_interactions, num_bins)

        # Assign users to a bin based on the number of items they interacted with
        bin_indices = np.digitize(log_values, bin_thresholds, right=True)

        # Create a dictionary that maps users to bins
        user_bin_dict = dict(zip(self.user_interactions_dict_train.keys(), bin_indices))
        return user_bin_dict
