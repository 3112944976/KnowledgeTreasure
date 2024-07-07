'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        # self.batch_size = args.batch_size
        self.norm_adj = norm_adj
        # 评估并保存层数
        self.layers = eval(args.layer_size)
        # 评估并保存衰减系数
        self.decay = eval(args.regs)[0]
        # Init the weight of user-item.
        self.embedding_dict, self.weight_dict = self.init_weight()
        # 将norm_adj转换为稀疏的torch张量，并移动到指定设备上
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # 使用Xavier初始化器，初始化嵌入层权重
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size)))
        })
        # 使用Xavier初始化器，初始化各嵌入传播层权重W1和W2
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
            功能：将稀疏矩阵norm_adj转换为稀疏的torch张量
            X示例：X = sp.coo_matrix(([3, 1, 4, 1], ([0, 1, 2, 2], [0, 1, 0, 2])), shape=(3, 3))
            return.shape == torch.Size([3, 3])
            return.to_dense() == [[3.0, 0., 0.], [0., 1., 0.], [4., 0., 1.]]
        """
        coo = X.tocoo()  # 将norm_adj转换为COO格式
        i = torch.LongTensor([coo.row, coo.col])  # 行和列索引转换为LongTensor
        v = torch.from_numpy(coo.data).float()  # 数据转换为FloatTensor
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        """
            功能：稀疏张量的dropout方法
        """
        random_tensor = 1 - rate  # 计算dropout保留概率
        random_tensor += torch.rand(noise_shape).to(x.device)  # 生成一个随机张量,并移动到指定设备
        dropout_mask = torch.floor(random_tensor).type(torch.bool)  # 将随机张量向下取整并转换为布尔型张量
        i = x._indices()  # 获取稀疏张量x的索引
        v = x._values()  # 获取稀疏张量x的值

        i = i[:, dropout_mask]  # 根据dropout_mask筛选索引
        v = v[dropout_mask]  # 根据dropout_mask筛选值
        # 根据筛选后的索引和值生成稀疏张量，并移动到指定设备
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        """
        功能：bpr损失函数
        Returns: 返回总损失、最大化损失和嵌入正则化损失
        """
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)  # 计算正样本得分
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)  # 计算负样本得分
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)  # 计算logsigmoid损失
        mf_loss = -1 * torch.mean(maxi)  # 计算最大化损失
        # 计算正则项
        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        # 计算嵌入正则化损失
        emb_loss = self.decay * regularizer / self.batch_size
        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        """
        功能：基于点积的评分函数
        """
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        # 1. 节点dropout：对邻接矩阵进行稀疏dropout
        A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout, self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        # 2. 初始化ego_embeddings，为用户和物品的嵌入向量拼接而成的张量E
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        # 初始化all_embeddings，将E^(0)放入其中
        all_embeddings = [ego_embeddings]
        # 3. 递归的嵌入传播学习
        for k in range(len(self.layers)):
            # side_embeddings=A_hat*E^(l-1), A_hat=node_dropout(norm_adj)
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            # A_hat*E^(l-1)*(W_1)^l
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]
            # A_hat*E^(l-1)*E^(l-1)
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # bi_embeddings*(W_2)^l
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]
            # 通过非线性激活，获得E^(l)
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            # 消息dropout：对E^(l)进行dropout操作
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            # 对E^(l)进行L2范数归一化
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
        # 4. 嵌入组合：将所有层的嵌入向量E，拼接成一个大的嵌入矩阵
        all_embeddings = torch.cat(all_embeddings, 1)  # 在维度1上拼接所有嵌入向量
        u_g_embeddings = all_embeddings[:self.n_user, :]  # 获取串联的用户嵌入向量
        i_g_embeddings = all_embeddings[self.n_user:, :]  # 获取串联的物品嵌入向量
        # 5. 根据输入的批次列表users、pos_items、neg_items，获得对应的嵌入表征
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
