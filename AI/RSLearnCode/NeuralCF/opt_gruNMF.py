import csv
import os
import heapq
import datetime
import numpy as np
import skopt.utils

import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import skopt.plots
from opt_NeuMF import to_named_params, load_results, save_results

import warnings
warnings.filterwarnings('ignore')


def get_train_instances(train, num_negatives):
    """用户打过分的为正样本，没打过分的为负样本。采样存储部分负样本"""
    np.random.seed(123)
    user_input, item_input, labels = [], [], []
    num_items = train.shape[1]
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


class MyGRU(Module):
    """GRU模块"""
    def __init__(self, seq_length: int, input_size: int, hidden_size: int):
        super().__init__()
        self.seq_length: int = seq_length
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.rnn = nn.GRUCell(self.input_size, self.hidden_size)
    def forward(self, X: torch.Tensor, h0=None):
        batch_size: int = X.shape[1]
        if h0 is None:
            prev_h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        else:
            prev_h = torch.squeeze(h0, 0)
        output = torch.zeros(self.seq_length, batch_size, self.hidden_size, device=X.device)  # 初始化output张量
        for i in range(self.seq_length):
            prev_h = self.rnn(X[i], prev_h)
            output[i] = prev_h
        return output, torch.unsqueeze(prev_h, 0)


class NeuralMFWithGRU(nn.Module):
    """NeuralMF with GRU"""
    def __init__(self, num_users, num_items, mf_dim, mlp_user_dim, mlp_item_dim, layers, dropout, hidden_size, num_layers=1):
        super(NeuralMFWithGRU, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)
        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mlp_user_dim)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mlp_item_dim)
        # # GRU组件：
        # self.gru = nn.GRU(input_size=mlp_user_dim+mlp_item_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.gru = MyGRU(1, mlp_user_dim+mlp_item_dim, hidden_size)  # 输入形状需为(seq_length, input_size, hidden_size)
        self.mlp_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.mlp_layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(p=dropout))
        self.linear2 = nn.Linear(2 * mf_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.long()
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])  # (100,8)(user_num,mf_dim)
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])  # (3706,8)(item_num,mf_dim)
        mf_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)  # (100,8)(user_num,mf_dim)
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[:, 0])  # (100,32)(user_num,mlp_user_dim)
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[:, 1])  # (3706,32)(item_num,mlp_item_dim)
        x = torch.cat([MLP_Embedding_User, MLP_Embedding_Item], dim=-1)  # (64,16)(batch_size,mlp_user_dim+mlp_item_dim)
        # print('x0', x.shape)
        # 升维
        x = x.unsqueeze(1)  # (batch_size,1,mlp_user_dim+mlp_item_dim)
        B, S, I = x.size()
        x = x.view(S, B, I)
        # print('x', x.shape)
        # gru_out, _ = self.gru(x)  # (batch_size,seq_len,hidden_num)
        gru_out, _ = self.gru(x)   # 输入形状需为(seq_length, batch_size, input_size)
        # print("gru0", gru_out.shape)
        # 降维
        # gru_out = gru_out[:, -1, :]  # (batch_size,hidden_num)
        gru_out = gru_out[-1, :, :]  # (batch_size,hidden_num)
        # print('gru', gru_out.shape)
        mlp_vec = F.relu(gru_out)
        for layer in self.mlp_layers:
            # print(mlp_vec.shape)
            mlp_vec = layer(mlp_vec)  # last_mlp_vec:(batch_size,layers[-1]:mf_dim)
        vector = torch.cat([mf_vec, mlp_vec], dim=-1)  # vector:(user_num,2*mf_dim)
        linear = self.linear2(vector)  # (100,1)(user_num,1)
        output = self.sigmoid(linear)  # (100,1)
        return output


def getHitRatio(ranklist, gtItem):
    """HitRation"""
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    """NDCG"""
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return np.log(2) / np.log(i + 2)
    return 0


def eval_one_rating(idx):
    """对testRatings中的一个用户样本进预测和评估"""
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    test_data = torch.tensor(np.vstack([users, np.array(items)]).T).to(device)
    predictions = _model(test_data)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].data.cpu().numpy()[0]
    items.pop()
    ranklist = heapq.nlargest(_K, map_item_score, key=lambda k: map_item_score[k])
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return hr, ndcg


def evaluate_model(model, testRatings, testNegatives, K):
    """整体上评估模型性能"""
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testNegatives = testNegatives
    _testRatings = testRatings
    _K = K
    hits, ndcgs = [], []
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return np.array(hits).mean(), np.array(ndcgs).mean()


# def train_and_eval(epochs, loss_func, optimizer, model, dl_train, topK, testRatings, testNegatives):
#     """考虑贝叶斯优化所需的返回值：当前参数组合下最佳ndcg"""
#     total_ndcgs = []
#     for epoch in range(epochs):
#         model.train()
#         for step, (features, labels) in enumerate(dl_train, 1):
#             features, labels = features.cuda(), labels.cuda()
#             optimizer.zero_grad()
#             predictions = model(features)
#             predictions = predictions.squeeze(1)
#             loss = loss_func(predictions, labels)
#             loss.backward()
#             optimizer.step()
#         model.eval()
#         hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK)
#         total_ndcgs.append(ndcgs)
#     max_ndcg = np.max(total_ndcgs)
#     return max_ndcg
#
#
# if __name__ == "__main__":
#     # 基本配置
#     batch_size = 128
#     np.random.seed(123)
#     file_path = './result/resultsGRU.pkl'
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # 参数定义
#     epochs = 12
#     num_layers = 1
#     num_negatives = 5
#     # 数据导入
#     train = np.load('ProcessedData/train.npy', allow_pickle=True).tolist()
#     testRatings = np.load('ProcessedData/testRatings.npy').tolist()
#     testNegatives = np.load('ProcessedData/testNegatives.npy').tolist()
#     # 切数据
#     # train = train[:500]
#     # testRatings = testRatings[:500]
#     # testNegatives = testNegatives[:500]
#     num_users, num_items = train.shape
#     # 搜索空间（固定：batch_size=128）
#     space = [
#         Integer(33, 50, name='topK'),
#         Integer(8, 40, name='mf_dim'),
#         Integer(1, 45, name='mlp_user_dim'),
#         Integer(1, 40, name='mlp_item_dim'),
#         # Integer(1, 5, name='num_negatives'),
#         Real(1e-4, 1e-1, prior='log-uniform', name='lr'),
#         # Integer(10, 20, name='epochs'),
#         Integer(40, 128, name='hidden_size'),
#         Real(0, 0.4, name='dropout'),
#         # Integer(1, 3, name='num_layers'),
#         Integer(4, 30, name='layers_1'),
#         Integer(4, 50, name='layers_2'),
#         Integer(4, 55, name='layers_3'),
#         Integer(220, 320, name='layers_4'),
#         Integer(4, 200, name='layers_5'),
#     ]
#     # 优化配置
#     HPO_PARAMS = {
#         'n_calls': 18,  # 总迭代次数
#         'n_random_starts': 1,
#         'base_estimator': 'ET',  # 指定用于拟合目标函数的基本估计器
#         'acq_func': 'gp_hedge',  # 采集函数，概率选择EI、PI和LCB中的一个
#         'random_state': 15
#     }
#     # 目标函数
#     @use_named_args(space)
#     def objective(topK, mf_dim, mlp_user_dim, mlp_item_dim, lr, hidden_size, dropout, layers_1, layers_2, layers_3, layers_4, layers_5):
#         """返回负的最大 NDCG 值，因为优化器的目标是最小化指标"""
#         # DataLoader
#         user_input, item_input, labels = get_train_instances(train, num_negatives)
#         train_x = np.vstack([user_input, item_input]).T
#         labels = np.array(labels)
#         train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(labels).float())
#         dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         # 自定义层结构
#         layers = [hidden_size, layers_1, layers_2, layers_3, layers_4, layers_5, mf_dim]
#         # 显示当前评估的参数组合
#         print(f"Evaluating: topK={topK}, mf_dim={mf_dim}, mlp_user_dim={mlp_user_dim}, mlp_item_dim={mlp_item_dim}, num_negatives={num_negatives}, lr={lr}, "
#               f"epochs={epochs}, hidden_size={hidden_size}, dropout={dropout}, num_layers={num_layers}, layers={layers}")
#         # 创建模型
#         model = NeuralMFWithGRU(num_users, num_items, mf_dim, mlp_user_dim, mlp_item_dim, layers, dropout, hidden_size, num_layers)
#         model.to(device)
#         # 模型训练与测试
#         loss_func = nn.BCELoss()
#         optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
#         max_ndcg = train_and_eval(epochs, loss_func, optimizer, model, dl_train, topK, testRatings, testNegatives)
#         return -max_ndcg
#     # 调参测试
#     old_results = load_results(file_path)
#     if old_results is not None:
#         print("继续测试")
#         print("old_results.x_iters: ", old_results.x_iters)
#         print("old_results.func_vals: ", old_results.func_vals)
#         results = gp_minimize(objective, space,
#                               x0=old_results.x_iters,
#                               y0=old_results.func_vals,
#                               **HPO_PARAMS)
#     else:
#         print("从零开始测试")
#         results = gp_minimize(objective, space, **HPO_PARAMS)
#     # 保存结果
#     save_results(results, file_path)
#     print("results.x_iters：", results.x_iters)
#     print("results.func_vals：", results.func_vals)
#     # 过程显示
#     skopt.plots.plot_objective(results)
#     plt.show()
#     # 打印最佳参数组合
#     best_params = to_named_params(results, space)
#     print("Best parameters:", best_params)

# 验证实验主函数
def train_and_eval(epochs, loss_func, optimizer, model, dl_train, topK, testRatings, testNegatives):
    test_csv_path = './result/test_results.csv'
    test_csv_file = open(test_csv_path, 'w', newline='')
    test_csv_writer = csv.writer(test_csv_file)
    test_csv_writer.writerow(['Epoch', 'LOSS', 'NDCG', 'HR'])
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for step, (features, labels) in enumerate(dl_train, 1):
            features, labels = features.cuda(), labels.cuda()
            optimizer.zero_grad()
            predictions = model(features)
            predictions = predictions.squeeze(1)
            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        model.eval()
        hr, ndcg = evaluate_model(model, testRatings, testNegatives, topK)
        test_csv_writer.writerow([epoch, loss_sum/step, ndcg, hr])
        info = (epoch, loss_sum/step, hr, ndcg)
        print(("\nEPOCH = %d, loss = %.3f, hr = %.3f, ndcg = %.3f") % info)
    test_csv_file.close()


if __name__ == "__main__":
    # 基本配置
    batch_size = 128
    np.random.seed(123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 参数设置
    topK = 40
    mf_dim = 38
    mlp_user_dim = 45
    mlp_item_dim = 3
    num_negatives = 5
    lr = 0.00439
    epochs = 15
    hidden_size = 62
    dropout = 0.3782
    num_layers = 1
    layers = [hidden_size, 11, 33, 55, 228, 177, mf_dim]
    # 数据导入
    train = np.load('ProcessedData/train.npy', allow_pickle=True).tolist()
    testRatings = np.load('ProcessedData/testRatings.npy').tolist()
    testNegatives = np.load('ProcessedData/testNegatives.npy').tolist()
    # 切数据
    # train = train[:100]
    # testRatings = testRatings[:100]
    # testNegatives = testNegatives[:100]
    num_users, num_items = train.shape
    # DataLoader
    user_input, item_input, labels = get_train_instances(train, num_negatives)
    train_x = np.vstack([user_input, item_input]).T
    labels = np.array(labels)
    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(labels).float())
    dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 模型实例化
    model = NeuralMFWithGRU(num_users, num_items, mf_dim, mlp_user_dim, mlp_item_dim, layers, dropout, hidden_size, num_layers)
    model.to(device)
    # print(model)
    # 模型训练与测试
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train_and_eval(epochs, loss_func, optimizer, model, dl_train, topK, testRatings, testNegatives)