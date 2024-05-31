import torch
import numpy as np
from mylightgcn.utils import tools
from tqdm import tqdm


def train_pairwise(samples, model, loss_class, optimizer, batch_size, device):
    model.train()
    samples = np.array(samples)
    # 从采样中提取用户、正样本物品、负样本物品，并转换为tensor对象
    users = torch.Tensor(samples[:, 0]).long().to(device)
    pos_items = torch.Tensor(samples[:, 1]).long().to(device)
    neg_items = torch.Tensor(samples[:, 2]).long().to(device)
    parameters_norm = torch.tensor(0).to(device)
    # 随机打乱用户、正样本物品和负样本物品的顺序
    users, pos_items, neg_items = tools.shuffle(users, pos_items, neg_items)
    # 计算总的批次数量
    total_batch = len(users) // batch_size + 1
    # 初始化平均损失值为0
    avg_loss = 0.
    # 使用自定义的minibatch将数据划分为若干批次数据，batch_i表示批次索引，tqdm用于显示训练进度
    for (batch_i, (batch_users, batch_pos, batch_neg)) in tqdm(enumerate(tools.minibatch(users, pos_items, neg_items, batch_size=batch_size)), desc="Training", total=total_batch, leave=False):
        optimizer.zero_grad()
        # 获取所有用户和物品的嵌入向量
        all_users_embeddings, all_items_embeddings = model()

        users_embeddings = all_users_embeddings[batch_users]
        pos_items_embeddings = all_items_embeddings[batch_pos]
        neg_items_embeddings = all_items_embeddings[batch_neg]

        users_embeddings_layer0 = model.users_embedding(batch_users)
        pos_items_embeddings_layer0 = model.items_embedding(batch_pos)
        neg_items_embeddings_layer0 = model.items_embedding(batch_neg)
        # 使用传入的损失函数，计算损失
        loss = loss_class(
            users_embeddings,
            pos_items_embeddings,
            neg_items_embeddings,
            users_embeddings_layer0,
            pos_items_embeddings_layer0,
            neg_items_embeddings_layer0,
            parameters_norm
        )
        loss.backward()
        optimizer.step()
        # 将当前批次的损失值累加到平均损失值中
        avg_loss += loss.item()
    # 计算平均损失值
    avg_loss = avg_loss / total_batch
    return avg_loss
