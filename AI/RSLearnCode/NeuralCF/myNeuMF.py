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


class NeuralMF(nn.Module):
    """base_NMF"""
    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NeuralMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)
        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], mf_dim)
        self.linear2 = nn.Linear(2 * mf_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        """base"""
        inputs = inputs.long()
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        mf_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[:, 0])
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[:, 1])
        x = torch.cat([MLP_Embedding_User, MLP_Embedding_Item], dim=-1)
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        mlp_vec = self.linear(x)
        vector = torch.cat([mf_vec, mlp_vec], dim=-1)
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


def train_and_eval(epochs, loss_func, optimizer, model, dl_train, topK, testRatings, testNegatives):
    """test"""
    test_csv_path = './result/test_results.csv'
    test_csv_file = open(test_csv_path, 'w', newline='')
    test_csv_writer = csv.writer(test_csv_file)
    test_csv_writer.writerow(['Epoch', 'LOSS', 'NDCG'])
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
        test_csv_writer.writerow([epoch, loss_sum/step, ndcg])
        info = (epoch, loss_sum/step, hr, ndcg)
        print(("\nEPOCH = %d, loss = %.3f, hr = %.3f, ndcg = %.3f") % info)
    test_csv_file.close()


if __name__ == "__main__":
    # 基本配置
    batch_size = 64
    np.random.seed(123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 参数设置
    topK = 33
    num_factors = 32
    num_negatives = 5
    lr = 0.0038
    epochs = 10
    layers = [num_factors * 2, 54, 122, 100]
    # 数据导入
    train = np.load('ProcessedData/train.npy', allow_pickle=True).tolist()
    testRatings = np.load('ProcessedData/testRatings.npy').tolist()
    testNegatives = np.load('ProcessedData/testNegatives.npy').tolist()
    num_users, num_items = train.shape
    # DataLoader
    user_input, item_input, labels = get_train_instances(train, num_negatives)
    train_x = np.vstack([user_input, item_input]).T
    labels = np.array(labels)
    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(labels).float())
    dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 模型实例化
    model = NeuralMF(num_users, num_items, num_factors, layers)
    model.to(device)
    # 模型训练与测试
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train_and_eval(epochs, loss_func, optimizer, model, dl_train, topK, testRatings, testNegatives)
