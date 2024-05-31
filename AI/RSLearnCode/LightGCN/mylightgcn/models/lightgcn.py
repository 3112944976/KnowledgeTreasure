import torch
from torch import nn
import torch.nn.functional as F
import torch.sparse as sparse


class LightGCN(nn.Module):
    """LightGCN模型实现类"""
    def __init__(self, users_num, items_num, embedding_dim, layers, keep_prob, adj_mat):
        super(LightGCN, self).__init__()
        self.users_num = users_num
        self.items_num = items_num
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.keep_prob = keep_prob
        self.adj_mat = adj_mat
        # 初始化用户和物品的嵌入层
        self.users_embedding = nn.Embedding(users_num, embedding_dim)
        self.items_embedding = nn.Embedding(items_num, embedding_dim)
        self.reset_parameters()
        # 初始化sigmoid输出层
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        """嵌入层权重初始化"""
        nn.init.normal_(self.users_embedding.weight, std=0.1)
        nn.init.normal_(self.items_embedding.weight, std=0.1)

    def forward(self):
        """
        adj_mat（torch.sparse.FloatTensor or torch.sparse.cuda.FloatTensor）: 邻接矩阵 (用户和物品的二部图表示)
        """
        # 将稀疏邻接矩阵转换为密集张量
        dense_adj_mat = self.adj_mat.to_dense()
        # 获取初始的嵌入结果emb
        user_emb = self.users_embedding.weight
        item_emb = self.items_embedding.weight
        emb = torch.cat([user_emb, item_emb], dim=0)
        # 节点表征的卷积聚合与传播
        embs = [emb]
        for _ in range(self.layers):
            dense_adj_mat = F.dropout(dense_adj_mat, p=self.keep_prob, training=self.training)
            emb = sparse.mm(dense_adj_mat, emb)
            embs.append(emb)
        # 嵌入组合各层生成的嵌入结果emb
        final_emb = torch.stack(embs, dim=1)
        final_emb = torch.mean(final_emb, dim=1)
        # 从final_emb中提取目标用户和物品的嵌入表征
        users_token, items_token = final_emb[:self.users_num], final_emb[self.users_num:]
        return users_token, items_token
    
    def get_user_rating(self, user_indices):
        """计算指定用户和所有项目间的预测分数"""
        users_token, items_token = self.forward()
        target_users_emb, items_emb = users_token[user_indices.long()], items_token
        scores = self.sigmoid(torch.matmul(target_users_emb, items_emb.t()))
        return scores
