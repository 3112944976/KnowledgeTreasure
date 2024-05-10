import torch
from datasets import Loader
import os
import world
import numpy as np
import math
from tqdm import tqdm

"""
计算嵌入向量的平滑性（smoothness）
"""


def compute_smoothness(graph, embeddings, type_, num_users):
    """
    Compute the smoothness of embeddings based on the given graph.
    Args:
        graph (list): 表示用户-物品交互的图
        embeddings (torch.Tensor): 用户和物品的嵌入向量
        type_ (str): T嵌入向量的类型，可以是 "users"（用户）或 "items"（物品）
        num_users (int): 用户的数量
    Returns:
        float: The smoothness of the embeddings.
    """
    # 初始化平滑性变量
    smoothness = 0
    # 根据嵌入向量类型确定起始和结束索引
    idx_start = 0 if type_ == "users" else num_users
    idx_end = num_users if type_ == "users" else len(embeddings)
    # 遍历嵌入向量中的每个用户或物品
    for u in tqdm(range(idx_start, idx_end)):
        for v in tqdm(range(idx_start, idx_end), leave=False):
            # 获取当前用户或物品的嵌入向量
            eu = embeddings[u]
            ev = embeddings[v]
            # 获取当前用户或物品的邻居节点索引
            neighbors_u = graph[u].coalesce().indices().squeeze()
            neighbors_v = graph[v].coalesce().indices().squeeze()
            # 计算当前用户或物品的邻居节点数量的平方根
            Nu = math.sqrt(neighbors_u.shape[0])
            Nv = math.sqrt(neighbors_v.shape[0])

            smoothness_strength = 0
            # 遍历当前用户或物品的邻居节点中的交集
            for i in np.intersect1d(neighbors_u.cpu(), neighbors_v.cpu()):
                # 获取邻居节点的邻居节点
                neighbors = graph[i].coalesce().indices().squeeze()
                # 计算邻居节点的数量
                num_neighbors = neighbors.shape[0]
                # 更新平滑性强度，即当前用户或物品与其邻居节点之间的平滑程度
                smoothness_strength += 1 / num_neighbors
            # 归一化平滑性强度，即：将平滑性强度除以当前用户或物品与其邻居节点数量的乘积
            smoothness_strength /= (Nu * Nv)
            # 计算平滑性，即:当前用户或物品与其邻居节点之间的平滑程度的加权平方误差之和
            smoothness += smoothness_strength * (eu - ev).pow(2).sum()
    return smoothness


if __name__ == "__main__":
    # 遍历所有嵌入向量文件
    for emb_file in os.listdir(world.EMBS_PATH):
        # 从文件名中解析出数据集名称、嵌入层次、嵌入方法等信息
        _, layer, method, dataset = emb_file.split("_")[: 4]
        # 加载数据集并获取稀疏图表示
        world.dataset = dataset
        dataloader = Loader(
            config=world.config, path=os.path.join("..", "data", dataset)
        )
        graph = dataloader.get_sparse_graph().to(world.device)
        num_users = dataloader.n_user
        # 加载嵌入向量并归一化
        emb_file_path = os.path.join(world.EMBS_PATH, emb_file)
        embeddings = torch.load(emb_file_path).to(world.device)
        embeddings /= torch.linalg.norm(embeddings, dim=1, ord=2).unsqueeze(-1)
        # 分别计算用户和物品的平滑性
        users_smoothness = compute_smoothness(
            graph, embeddings, type_="users", num_users=num_users)
        items_smoothness = compute_smoothness(
            graph, embeddings, type_="items", num_users=num_users)
        # 将结果写入到文件 "smoothness_results.txt" 中
        with open("smoothness_results.txt", "a") as w:
            w.write("users")
            w.write(" ")
            w.write(method)
            w.write(" ")
            w.write(dataset)
            w.write(" ")
            w.write(str(round(users_smoothness.item(), 2)))
            w.write("\n")
            w.write("items")
            w.write(" ")
            w.write(method)
            w.write(" ")
            w.write(dataset)
            w.write(" ")
            w.write(str(round(items_smoothness.item(), 2)))
            w.write("\n")
