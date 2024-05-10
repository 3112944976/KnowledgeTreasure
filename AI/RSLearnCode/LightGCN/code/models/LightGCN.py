from .BasicModel import BasicModel
from ..datasets import BasicDataset
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import torch


# 对稀疏矩阵进行dropout处理
class SparseDropout(nn.Module):
    def __init__(self, p):  # 接受一个表示dropout概率的参数p
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()  # 将输入的稀疏矩阵压缩
        drop_val = F.dropout(input_coal._values(), self.p, self.training)  # 对压缩后的值进行dropout处理
        # 返回基于pytorch的稀疏张量，形状同输入相同
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


# 处理稠密和稀疏矩阵的dropout操作
class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        # 根据输入的稠密或稀疏性，选择对应的dropout操作，并返回处理后的结果。
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


# 将稀疏矩阵转换为pytorch的稀疏张量
def sparse_matrix_to_torch(X):
    coo = X.tocoo()  # 将稀疏矩阵X转换为COO格式
    indices = np.array([coo.row, coo.col])  # 提取行、列索引，以及数据值
    # 返回一个基于pytorch的稀疏张量
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices),
        torch.FloatTensor(coo.data),
        coo.shape)


# 计算归一化的邻接矩阵，接受一个稀疏或稠密矩阵
def calc_A_hat(adj_matrix):
    adj_matrix = sp.csr_matrix(adj_matrix.cpu().to_dense().numpy())  # 转化为CSR格式
    nnodes = adj_matrix.shape[0]  # 获取图中的节点数，即：邻接矩阵的行数
    A = adj_matrix + sp.eye(nnodes)  # 邻接矩阵+单位矩阵（自连接）
    D_vec = np.sum(A, axis=1).A1  # 获取度矩阵
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)  # 获取度矩阵倒数的平方根
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)  # 创建一个对角矩阵
    return D_invsqrt_corr @ A @ D_invsqrt_corr


# 实现Personalized PageRank的迭代算法
class PPRPowerIteration(nn.Module):
    """
    接收：
        adg_matrix：邻接矩阵 类型为sp.spmatrix（表示图的结构）
        alpha：随机游走概率
        niter：迭代次数
        drop_prob：可选参数，dropout的概率
    """
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter
        # 获取归一化邻接矩阵：M
        M = calc_A_hat(adj_matrix)
        # 获取(1-a)*M：A_hat，并注册为模型的缓存区，使其在GPU和CPU上均能高效处理
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))
        # 若drop_prob不为None或0，则创建MixedDropout实例用于dropout
        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, E: torch.FloatTensor):
        """
        接收：
            E：初始归一化邻接矩阵
        """
        print('')
        preds = E  # 将E作为预测矩阵的起始值
        for _ in range(self.niter):  # 进行niter次迭代
            A_drop = self.dropout(self.A_hat)  # 对A_hat进行dropout处理
            preds = A_drop @ preds + self.alpha * E  # 核心公式
        return preds


# LightGCN实现类
class LightGCN(BasicModel):
    """
        config (dict): 模型的配置参数
        dataset (BasicDataset): 包含用户-项目交互的数据集。
        num_users (int): 数据集中唯一用户数
        num_items (int): 数据集中唯一项目数
        latent_dim (int): 嵌入维数
        embs (torch.Tensor or None): 嵌入矩阵的形状：(num_users + num_items, latent_dim) or None.
        n_layers (int): LightGCN的层数
        keep_prob (float): Dropout的保持概率
        a_split (bool): 是否拆分邻接矩阵
        embedding_user (torch.nn.Embedding): 用户嵌入层
        embedding_item (torch.nn.Embedding): 项目嵌入层
        sigmoid (torch.nn.Sigmoid): Sigmoid activation function.
        graph (torch.sparse.FloatTensor): 稀疏图表示
    """
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()  # 调用父类
        self.config = config  # 实例化输入的配置参数
        self.dataset = dataset  # 实例化输入的数据集
        self.embs = None  # 初始化嵌入矩阵
        self.__init_weight()

    def get_embedding_matrix(self):
        """获取嵌入矩阵：(num_users + num_items, latent_dim) or None"""
        return self.embs

    def __init_weight(self):  # 初始化用户和项目的嵌入权重
        self.num_users = self.dataset.n_users  # 获取数据集中的唯一用户数
        self.num_items = self.dataset.m_items  # 获取数据集中的唯一项目数
        self.latent_dim = self.config["latent_dim_rec"]  # 从配置参数中获取嵌入维数
        self.n_layers = self.config["lightGCN_n_layers"]  # 从配置参数中获取LightGCN层数
        self.keep_prob = self.config["keep_prob"]  # 从配置参数中获取dropout的保持概率
        self.a_split = self.config["A_split"]  # 从配置参数中获取是否拆分邻接矩阵的标志
        # 创建用户嵌入层
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # 创建项目嵌入层
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 若不使用预训练数据，则使用正态分布随机初始化嵌入权重
        if self.config["pretrain"] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print("Using NORMAL distribution initializer.")
        else:  # 否则，从配置参数中加载预训练的用户和项目嵌入权重
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config["user_emb"]))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config["item_emb"]))
            print("Using pretrained data.")

        self.sigmoid = nn.Sigmoid()  # 创建sigmoid激活函数
        self.graph = self.dataset.get_sparse_graph()  # 获取数据集的稀疏图表示
        self.embs = None  # 存储节点的嵌入表示
        if self.config['model'] == 'appnp':  # 若配置参数中指定的模型为APPNP
            self.propagation = PPRPowerIteration(self.graph, alpha=self.config['alpha'], niter=self.config['num_walks'])
        print(f"LightGCN is ready to go (dropout: {self.config['dropout']}).")

    @staticmethod
    def __dropout_x(x, keep_prob):
        """
        对稀疏张量x按照keep_prob的保持概率，进行dropout处理
        Args:
            x (torch.sparse.FloatTensor): Sparse tensor to apply dropout on.
            keep_prob (float): Dropout keep probability.
        Returns:
            torch.sparse.FloatTensor: The dropout-applied sparse tensor.
        """
        size = x.size()  # 获取稀疏张量的尺寸
        index = x.indices().t()  # 获取稀疏张量中非零值的索引，并转置（二维数组）
        values = x.values()  # 获取稀疏张量的非零值（一维数组）
        # 生成与值数量相同的随机数并加上保持概率
        random_index = torch.rand(len(values)) + keep_prob
        # 将随机数转换为整数类型，并转换为布尔型，以便进行索引选择
        random_index = random_index.int().bool()
        # 选择保持的非零元素的索引
        index = index[random_index]
        # 针对保持的非零元素取值，通过除以keep_prob进行缩放，以保持期望值不变。
        values = values[random_index] / keep_prob
        # 构建新的稀疏张量，重塑为同原始稀疏张量一致的尺寸
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        """将Dropout应用于图表示"""
        if self.a_split:  # 若需要拆分邻接矩阵
            graph = []  # 存储拆分后的图表示
            for g in self.graph:  # 对邻接矩阵列表中的每个邻接矩阵进行dropout处理并存储
                graph.append(self.__dropout_x(g, keep_prob))
        else:  # 若无需拆分邻接矩阵
            graph = self.__dropout_x(self.graph, keep_prob)
        return graph

    def forward(self):
        users_emb = self.embedding_user.weight  # 获取用户嵌入权重
        items_emb = self.embedding_item.weight  # 获取项目嵌入权重
        all_emb = torch.cat([users_emb, items_emb])  # 将用户和项目嵌入权重连接起来
        embs = [all_emb]  # 引入初始嵌入权重矩阵，初始化一个嵌入列表
        if self.config["dropout"]:  # 若配置参数中启用了Dropout
            if self.training:  # 若处于训练阶段
                print("Dropping.")
                g_dropped = self.__dropout(self.keep_prob)
            else:  # 若处于评估阶段
                g_dropped = self.graph
        else:  # 若未启用Dropout
            g_dropped = self.graph

        for _ in range(self.n_layers):
            if self.a_split:  # 若需要拆分邻接矩阵
                temp_emb = []  # 初始化一个临时嵌入列表
                for f in range(len(g_dropped)):  # 遍历每个拆分后的邻接矩阵
                    # 将子邻接矩阵同嵌入向量相乘，并存储结果
                    temp_emb.append(torch.sparse.mm(g_dropped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)  # 将临时嵌入列表中的所有嵌入向量连接起来
                all_emb = side_emb  # 更新all_emb
            else:  # 若无需拆分邻接矩阵
                all_emb = torch.sparse.mm(g_dropped, all_emb)
            embs.append(all_emb)  # 记忆每次聚合卷积后的嵌入权重
        embs = torch.stack(embs, dim=1)  # 将嵌入矩阵列表中的所有嵌入向量堆叠起来，形成嵌入矩阵。
        # 若配置参数中指定要保存嵌入向量
        if self.config["save_embs"]:
            self.embs = embs
        # 若配置参数中指定要使用单个嵌入向量
        if self.config["single"]:
            light_out = embs[:, -1, :].squeeze()  # 选择最后一个嵌入向量并压缩维度
        else:
            light_out = torch.mean(embs, dim=1)  # 否则，使用平均嵌入向量，即：嵌入组合环节
        # 若配置参数中指定使用APPNP模型
        if self.config['model'] == 'appnp':
            # Approximate personalized propagation of neural predictions
            light_out = self.propagation(light_out)
        # 将结果拆分为用户和项目的嵌入向量
        all_users_embeddings, all_items_embeddings = torch.split(light_out, [self.num_users, self.num_items])
        return all_users_embeddings, all_items_embeddings

    def get_user_rating(self, users):
        """
        用于获取给定用户的预测评分
        Args:
            users (torch.Tensor): Tensor containing user indices.
        Returns:
            torch.Tensor: Predicted ratings for the users, with values between 0 and 1.
        """
        all_users, all_items = self.forward()  # 获取所有用户和项目的嵌入向量
        users_emb = all_users[users.long()]  # 获取指定用户的嵌入向量
        items_emb = all_items  # 获取所有项目的嵌入向量
        # 计算指定用户和所有项目间的预测分数
        rating = self.sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating