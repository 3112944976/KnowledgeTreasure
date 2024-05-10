import torch
import utils
import world
from utils import Timer
from tqdm import tqdm


def train_pairwise(dataset, model, loss_class, optimizer):
    """
    Train the model using pairwise ranking loss.
    Returns:
        tuple: A tuple containing the average loss value and timing info.
    """
    model.train()
    # 计时器，并从数据集中均匀地采样数据
    with Timer(name="Sample"):
        samples = utils.uniform_sample_original(dataset)
    # 从采样中提取用户、正样本物品、负样本物品，并转换为tensor对象
    users = torch.Tensor(samples[:, 0]).long()
    pos_items = torch.Tensor(samples[:, 1]).long()
    neg_items = torch.Tensor(samples[:, 2]).long()
    # 将数据移动到GPU或CPU上
    users = users.to(world.device)
    pos_items = pos_items.to(world.device)
    neg_items = neg_items.to(world.device)
    # 随机打乱用户、正样本物品和负样本物品的顺序
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    # 计算总的批次数量
    total_batch = len(users) // world.config["batch_size"] + 1
    # 初始化平均损失值为0
    avg_loss = 0.
    # 使用自定义的minibatch将数据划分为若干批次数据，batch_i表示批次索引，tqdm用于显示训练进度
    for (batch_i, (batch_users, batch_pos, batch_neg)) \
            in tqdm(enumerate(utils.minibatch(users, pos_items, neg_items, batch_size=world.config["batch_size"])), desc="Training", total=total_batch, leave=False):
        optimizer.zero_grad()
        # 获取所有用户和物品的嵌入向量
        all_users_embeddings, all_items_embeddings = model()

        users_embeddings = all_users_embeddings[batch_users]
        pos_items_embeddings = all_items_embeddings[batch_pos]
        neg_items_embeddings = all_items_embeddings[batch_neg]

        users_embeddings_layer0 = model.embedding_user(batch_users)
        pos_items_embeddings_layer0 = model.embedding_item(batch_pos)
        neg_items_embeddings_layer0 = model.embedding_item(batch_neg)
        # 使用传入的损失函数，计算损失
        loss = loss_class(
            users_embeddings,
            pos_items_embeddings,
            neg_items_embeddings,
            users_embeddings_layer0,
            pos_items_embeddings_layer0,
            neg_items_embeddings_layer0,
            model.parameters_norm()
        )
        loss.backward()
        optimizer.step()
        # 将当前批次的损失值累加到平均损失值中
        avg_loss += loss.item()
    # 计算平均损失值
    avg_loss = avg_loss / total_batch
    # 获取计时器的信息
    time_info = Timer.dict()
    # 重置计时器
    Timer.zero()
    return avg_loss, time_info
