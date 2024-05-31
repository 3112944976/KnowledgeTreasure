import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.sparse as sparse
import torch.nn.functional as F

from loss import BPRLoss
from models import lightgcn
from DataLoader import Loader
from procedures import train_pairwise, eval_pairwise

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    # 初始设置
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_data_path = "amazon-electro/train.txt"
    test_data_path = "amazon-electro/test.txt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 参数定义
    topk = 20
    layers = 4
    epochs = 500
    keep_prob = 0.1
    embedding_dim = 64
    batch_size = 2048
    test_batch_size = 100
    lr = 0.001
    weight_decay = 1e-4
    # 1. 加载数据
    adj_mat, samples, users_num, items_num, train_dict = Loader.train_loader(train_data_path)
    test_dict = Loader.test_loader(test_data_path)
    adj_mat = adj_mat.to(device)
    # 2. 加载模型
    model = lightgcn.LightGCN(users_num, items_num, embedding_dim, layers, keep_prob, adj_mat)
    model.to(device)
    # 3. 加载损失函数和优化器
    loss_fun = BPRLoss.BPRLoss(weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 4. 模型训练与测试
    losses, precisions, recalls, ndcgs, metrics_epochs = [], [], [], [], []
    for epoch in range(epochs):
        avg_loss = train_pairwise.train_pairwise(samples, model, loss_fun, optimizer, batch_size, device)
        losses.append(avg_loss)
        print("epochs:{0} loss:{1}".format(epoch, avg_loss))
        if epoch % 10 == 0 and epoch != 0:
            precision, recall, ndcg = eval_pairwise.eval_pairwise(train_dict, test_dict, model, test_batch_size, topk, device)
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
            metrics_epochs.append(epoch)
    # 5. 结果可视化
    fig, ax1 = plt.subplots()
    # 绘制损失折线
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(epochs), losses, color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)
    # 绘制测试度量折线
    ax2 = ax1.twinx()
    ax2.set_ylabel('Metrics', color='tab:blue')
    ax2.plot(metrics_epochs, precisions, 'bo-', label='Precision', markersize=3)
    ax2.plot(metrics_epochs, recalls, 'gx-', label='Recall', markersize=3)
    ax2.plot(metrics_epochs, ndcgs, 'ms-', label='NDCG', markersize=3)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # 标注最大数据点
    max_precision = max(precisions)
    max_recall = max(recalls)
    max_ndcg = max(ndcgs)
    max_precision_index = precisions.index(max_precision)
    max_recall_index = recalls.index(max_recall)
    max_ndcg_index = ndcgs.index(max_ndcg)
    ax2.text(metrics_epochs[max_precision_index], max_precision + 0.002, f'{max_precision:.4f}', color='blue', ha='center')
    ax2.text(metrics_epochs[max_recall_index], max_recall + 0.002, f'{max_recall:.4f}', color='green', ha='center')
    ax2.text(metrics_epochs[max_ndcg_index], max_ndcg + 0.002, f'{max_ndcg:.4f}', color='magenta', ha='center')
    # 图表设置
    fig.legend(loc='center')
    plt.title('Training Loss and Performance Metrics Over Epochs')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
