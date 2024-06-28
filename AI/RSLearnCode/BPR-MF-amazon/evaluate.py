import numpy as np
import torch


# 判断标签物品gt_item是否在推荐列表pred_items中
def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


# 计算标签物品gt_item在推荐列表pred_items中的归一化折损累计增益（NDCG）。
def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


# 根据给定的模型和测试数据加载器，进行评估
def metrics(model, test_loader, top_k):
	HR, NDCG = [], []
	# 读取test_loader中当前批次下的三个列表user item_i item_j。比如，user=[2,2,2,..., 2]，len(user)=100
	for user, item_i, item_j in test_loader:
		user = user.cuda()
		item_i = item_i.cuda()
		item_j = item_j.cuda()  # 在测试时，item_j无用。
		# 计算属于用户u的100个物品的预测值（1个正例+99个负例）
		prediction_i, prediction_j = model(user, item_i, item_j)
		# 从prediction_i中取出分数最高的top_k个物品的索引
		_, indices = torch.topk(prediction_i, top_k)
		# 1. 从item_i列表中取出推荐物品ID，并存储到列表对象recommends中
		recommends = torch.take(item_i, indices).cpu().numpy().tolist()
		# 2. 获取标签物品gt_item
		gt_item = item_i[0].item()
		# 3. 度量计算
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
	return np.mean(HR), np.mean(NDCG)
