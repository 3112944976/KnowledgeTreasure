import torch


class BPRLoss(object):
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay

    def __call__(self,
                 users_embeddings,
                 pos_items_embeddings,
                 neg_items_embeddings,
                 users_embeddings_layer0,
                 pos_items_embeddings_layer0,
                 neg_items_embeddings_layer0,
                 parameters_norm):
        # 计算正则化损失：使用 L2 范数对每个用户、正样本和负样本的嵌入向量进行正则化。
        # 然后，除以嵌入向量的数量，以便损失值不会受到批量大小的影响。
        reg_loss = (1 / 2) * (users_embeddings_layer0.norm(2).pow(2) +
                              pos_items_embeddings_layer0.norm(2).pow(2) +
                              neg_items_embeddings_layer0.norm(2).pow(2) +
                              parameters_norm
                              ) / users_embeddings.shape[0]
        # 计算正样本的得分，即用户嵌入向量和正样本嵌入向量的点积。
        pos_scores = torch.mul(users_embeddings, pos_items_embeddings)
        pos_scores = torch.sum(pos_scores, dim=1)
        # 计算负样本的得分，即用户嵌入向量和负样本嵌入向量的点积。
        neg_scores = torch.mul(users_embeddings, neg_items_embeddings)
        neg_scores = torch.sum(neg_scores, dim=1)
        # 计算损失。使用Softplus函数来转换负样本得分和正样本得分的差值，确保损失值始终为正。
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # 将正则化损失乘以权重衰减系数，并加到总损失上。
        reg_loss *= self.weight_decay
        loss += reg_loss
        return loss
