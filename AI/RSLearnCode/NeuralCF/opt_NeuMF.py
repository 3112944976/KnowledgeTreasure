import csv
import os
import heapq
import datetime
import numpy as np
import skopt.utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import skopt.plots

import warnings
warnings.filterwarnings('ignore')


def get_train_instances(train, num_negatives):
    """用户打过分的为正样本，没打过分的为负样本。采样存储部分负样本"""
    np.random.seed(123)
    user_input, item_input, labels = [], [], []
    num_items = train.shape[1]
    for (u, i) in train.keys():  # 只遍历存在评分的用户-物品对：994169对
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instance
        for t in range(num_negatives):  # 每个正例样本，都追加num_negatives个负例样本，到训练集中
            j = np.random.randint(num_items)
            while (u, j) in train:  # 确保随机选择的负例物品不在训练集中
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


class NeuralMF(nn.Module):
    """base_NMF"""
    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NeuralMF, self).__init__()
        # GMF部分的用户和物品Embedding层
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)
        # MLP部分的用户和物品Embedding层
        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
        # MLP部分的隐藏及输出层
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], mf_dim)
        # NeuMF模型的输出层
        self.linear2 = nn.Linear(2 * mf_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """base"""
        # 将批次数据inputs，输入转换为long类型
        inputs = inputs.long()
        # GMF部分的用户和物品的embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        # 将用户和物品的嵌入表示逐元素相乘
        mf_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)

        # MLP部分的用户和物品的embedding
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[:, 0])
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[:, 1])
        # 拼接用户和物品的嵌入表示
        x = torch.cat([MLP_Embedding_User, MLP_Embedding_Item], dim=-1)
        # 通过循环完成多层线性层的学习
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        mlp_vec = self.linear(x)
        # 拼接GMF和MLP部分的输出
        vector = torch.cat([mf_vec, mlp_vec], dim=-1)
        # 输出层
        linear = self.linear2(vector)
        output = self.sigmoid(linear)
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
    print(predictions.shape)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].data.cpu().numpy()[0]
    items.pop()
    ranklist = heapq.nlargest(_K, map_item_score, key=lambda k: map_item_score[k])  # heapq是堆排序算法， 取前K个
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
    hits, ndcgs = [], []  # 存储所有评分的Hit Ratio和NDCG
    for idx in range(len(_testRatings)):  # 遍历所有测试评分
        (hr, ndcg) = eval_one_rating(idx)  # 对每个评分进行评估
        hits.append(hr)
        ndcgs.append(ndcg)
    # return hr, ndcg
    return np.array(hits).mean(), np.array(ndcgs).mean()


def train_and_eval(epochs, loss_func, optimizer, model, dl_train, testRatings, testNegatives, topK):
    """版本1：考虑贝叶斯优化所需的返回值：当前参数组合下最佳ndcg"""
    # total_loss = 0.0
    # total_hits = []
    total_ndcgs = []

    for epoch in range(epochs):
        model.train()
        # epoch_loss = 0.0
        for step, (features, labels) in enumerate(dl_train, 1):
            features, labels = features.cuda(), labels.cuda()
            optimizer.zero_grad()
            predictions = model(features)
            predictions = predictions.squeeze(1)
            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
        #     epoch_loss += loss.item()
        #
        # total_loss += epoch_loss

        model.eval()
        hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK)
        # total_hits.append(hits)
        total_ndcgs.append(ndcgs)

    # avg_loss = total_loss / epochs
    # max_hr = np.max(total_hits)
    max_ndcg = np.max(total_ndcgs)
    return max_ndcg


def to_named_params(results, search_space):
    """skopt中用于输出最佳参数"""
    params = results.x
    param_dict = {}
    params_list  =[(dimension.name, param) for dimension, param in zip(search_space, params)]
    for item in params_list:
        param_dict[item[0]] = item[1]
    return(param_dict)


def load_results(file_path):
    """加载结果"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return skopt.load(f)
    else:
        return None


def save_results(results, file_path):
    """保存结果"""
    with open(file_path, 'wb') as f:
        skopt.dump(results, f)


#  版本2：考虑存储历史最佳参数
if __name__ == "__main__":
    # 基本配置
    batch_size = 128
    np.random.seed(123)
    file_path = './result/results.pkl'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据导入
    train = np.load('ProcessedData/train.npy', allow_pickle=True).tolist()
    testRatings = np.load('ProcessedData/testRatings.npy').tolist()
    testNegatives = np.load('ProcessedData/testNegatives.npy').tolist()
    # 切片数据
    train = train[:500]
    testRatings = testRatings[:500]
    testNegatives = testNegatives[:500]
    num_users, num_items = train.shape
    # 搜索空间（固定：batch_size=128）
    space = [
        Integer(1, 80, name='topK'),
        Integer(4, 100, name='num_factors'),
        Integer(4, 100, name='num_negatives'),
        Real(1e-4, 1e-1, prior='log-uniform', name='lr'),
        Integer(10, 50, name='epochs'),
        Integer(8, 256, name='layers_1'),
        Integer(8, 256, name='layers_2'),
        Integer(8, 256, name='layers_3'),
    ]
    # 优化配置
    HPO_PARAMS = {
        'n_calls': 2,  # 总迭代次数
        'n_random_starts': 1,
        'base_estimator': 'ET',  # 指定用于拟合目标函数的基本估计器
        'acq_func': 'gp_hedge',  # 采集函数，概率选择EI、PI和LCB中的一个
        'random_state': 15
    }
    # 目标函数
    @use_named_args(space)
    def objective(topK, num_factors, num_negatives, lr, epochs, layers_1, layers_2, layers_3):
        """返回负的最大 NDCG 值，因为优化器的目标是最小化指标"""
        # DataLoader
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        train_x = np.vstack([user_input, item_input]).T
        labels = np.array(labels)
        train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(labels).float())
        dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 自定义层结构
        layers = [num_factors * 2, layers_1, layers_2, layers_3]
        # layers = [num_factors * 2, 54, 122, layers_3]
        # 显示当前评估的参数组合
        print(f"Evaluating: topK={topK}, num_factors={num_factors}, num_negatives={num_negatives}, lr={lr}, epochs={epochs}, layers={layers}")
        # 创建模型
        model = NeuralMF(num_users, num_items, num_factors, layers)
        model.to(device)
        # 模型训练与测试
        loss_func = nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        max_ndcg = train_and_eval(epochs, loss_func, optimizer, model, dl_train, testRatings, testNegatives, topK)
        return -max_ndcg
    # 调参测试
    old_results = load_results(file_path)
    if old_results is not None:
        print("继续测试")
        print("old_results.x_iters: ", old_results.x_iters)
        print("old_results.func_vals: ", old_results.func_vals)
        results = gp_minimize(objective, space,
                              x0=old_results.x_iters,
                              y0=old_results.func_vals,
                              **HPO_PARAMS)
    else:
        print("从零开始测试")
        results = gp_minimize(objective, space, **HPO_PARAMS)
    # 保存结果
    save_results(results, file_path)
    print("results.x_iters：", results.x_iters)
    print("results.func_vals：", results.func_vals)
    # 过程显示
    skopt.plots.plot_objective(results)
    plt.show()
    # 打印最佳参数组合
    best_params = to_named_params(results, space)
    print("Best parameters:", best_params)
