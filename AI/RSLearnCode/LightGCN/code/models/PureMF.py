"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
from .BasicModel import BasicModel
from torch import nn
import torch


class PureMF(BasicModel):
    """
    Pure Matrix Factorization (PureMF) model.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (Dataset): The dataset containing user-item interactions.

    Attributes:
        num_users (int): Number of unique users in the dataset.
        num_items (int): Number of unique items in the dataset.
        latent_dim (int): Dimensionality of the latent space.
        sigmoid (torch.nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self, config, dataset):
        """
        Initialize the PureMF model.

        Args:
            config (dict): Configuration parameters for the model.
            dataset (Dataset): The dataset containing user-item interactions.
        """
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users  # 初始化了用户和物品数及嵌入维数
        self.num_items = dataset.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.sigmoid = nn.Sigmoid()
        self.__init_weight()  # 初始化嵌入矩阵权重

    def get_embedding_matrix(self):
        """
        获取用户和物品的嵌入矩阵
        Returns:
            torch.Tensor: The embedding matrix of shape
                          (num_users + num_items, 1, latent_dim).
        """
        return torch.vstack((
            self.embedding_user.weight.data,
            self.embedding_item.weight.data
        )).unsqueeze(1)

    def __init_weight(self):
        """
        初始化用户和物品嵌入矩阵权重，使其服从N(0, 1)的正态分布。
        """
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("Using Normal distribution N(0, 1) initialization for PureMF.")

    def get_user_rating(self, users):
        """
        给定用户列表，获取对应用户的预测评分

        Args:
            users (torch.Tensor): Tensor containing user indices.

        Returns:
            torch.Tensor: Predicted ratings for the users, 
                          with values between 0 and 1.
        """
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())

        return self.sigmoid(scores)

    def forward(self):
        """
        Forward pass of the PureMF model.

        Returns:
            tuple: A tuple of user and item embedding weights.
        """
        return self.embedding_user.weight, self.embedding_item.weight
