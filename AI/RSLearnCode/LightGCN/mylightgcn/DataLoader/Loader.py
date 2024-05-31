import torch
import numpy as np


def train_loader(file_path):
    # 1. 读取数据文件
    file_path = "../data/" + str(file_path)
    data = []
    train_inter_num = 0
    with open(file_path, 'r') as file:
        for line in file:
            data.append(list(map(int, line.strip().split())))
    # 2. 创建用户-物品交互字典user_item_dict
    user_item_dict = {}
    for idx, line in enumerate(data):
        user_id = idx
        item_ids = line[1:]
        train_inter_num += len(item_ids)  # 累加计算训练数据中的总交互数
        user_item_dict[user_id] = item_ids
    # 3. 获取总用户数num_users和总物品数num_items
    num_users = len(user_item_dict)
    num_items = max(max(item_ids) for item_ids in user_item_dict.values()) + 1
    # 4. 创建稀疏矩阵adj_mat
    rows = []  # 存储稀疏矩阵的行索引
    cols = []  # 存储稀疏矩阵的列索引
    for user_id, item_ids in user_item_dict.items():
        rows.extend([user_id] * len(item_ids))
        cols.extend(item_ids)
    values = [1] * len(rows)  # 存储稀疏矩阵的值。
    # torch.sparse_coo_tensor()：参数1是非零元素的坐标，参数2是非零元素的值，参数3是稀疏矩阵的形状
    adj_mat = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(values), (num_users, num_items)).to_sparse().to(torch.float32)

    # 5. 获取原始邻接矩阵R（将稀疏张量adj_mat转换为密集张量R）
    R = adj_mat.to_dense()
    # 6. 构建对称邻接矩阵A=[[0, R], [R^T, 0]]
    top_row = torch.cat((torch.zeros((R.size(0), R.size(0))), R), dim=1)
    bottom_row = torch.cat((R.t(), torch.zeros((R.size(1), R.size(1)))), dim=1)
    A = torch.cat((top_row, bottom_row), dim=0)
    # 7. 计算A的度矩阵D
    D = torch.sum(A, dim=1)
    # 8. L2归一化处理
    D_sqrt_inv = torch.diag(torch.pow(D, -0.5))
    A_normalized = torch.mm(torch.mm(D_sqrt_inv, A), D_sqrt_inv)
    # 9. 创建稀疏张量
    nonzero_indices = torch.nonzero(A_normalized)  # 获取非零元素的索引和值
    nonzero_values = A_normalized[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    A_normalized_sparse = torch.sparse_coo_tensor(nonzero_indices.t(), nonzero_values, A_normalized.size())

    # 10. 样本采样
    users = np.random.randint(0, num_users, train_inter_num)
    samples = []
    for user in users:
        pos_item_list = user_item_dict[user]
        if len(pos_item_list) == 0:
            continue
        # 针对每个可重复的用户ID，随机采样一个正例样本pos_item
        pos_index = np.random.randint(0, len(pos_item_list))
        pos_item = pos_item_list[pos_index]
        # 循环寻找该用户的一个负样本neg_item
        while True:
            neg_item = np.random.randint(0, num_items)
            if neg_item in pos_item_list:
                continue
            else:
                break
        # 样本保存：将每个样本（用户、正样本、负样本）存储在samples列表中
        samples.append([user, pos_item, neg_item])
    return A_normalized_sparse, samples, num_users, num_items, user_item_dict


def test_loader(file_path):
    # 1. 读取数据文件
    file_path = "../data/" + str(file_path)
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(list(map(int, line.strip().split())))
    # 2. 创建用户-物品交互字典user_item_dict
    user_item_dict = {}
    for idx, line in enumerate(data):
        user_id = idx
        item_ids = line[1:]
        user_item_dict[user_id] = item_ids
    return user_item_dict
