import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.01, 
	help="learning rate")
parser.add_argument("--lamda", 
	type=float, 
	default=0.001, 
	help="model regularization rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=4096, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=50,
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
# len(train_data)=len(train_mat)
# train_data==test_data==[[user_id, item_id], ...]
# train_mat== ((user_id, item_id) 1) ...
# 1. 加载数据集
train_path = "../data/amazon-electro/train.txt"
test_path = "../data/amazon-electro/test.txt"
train_data, test_data, user_num, item_num, train_mat = data_utils.load_all(train_path, test_path, args.test_num_ng)

# 2. 构建训练和测试数据迭代器
train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.BPRData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)  # 修改，原本为num_workers=4
# 由于test.negative中每行代表一个用户，每行中第一个元素(u,i)中的i为标签物品，剩下99个元素i中的i皆为负例物品。共100个，因此batch_size必须为test_num_ng+1，即：100。
test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
# 3. 加载模型和优化器
model = model.BPR(user_num, item_num, args.factor_num)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamda)
# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
# 4. 模型训练
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train()
	start_time = time.time()
	total_loss = 0
	# 在每个epoch开始时对训练数据集进行负样本采样
	train_loader.dataset.ng_sample()
	# user为批次用户列表，item_i为批次正例物品列表（依次对应相同位置的用户），item_j为批次负例物品列表
	for user, item_i, item_j in train_loader:
		user = user.cuda()
		item_i = item_i.cuda()
		item_j = item_j.cuda()
		model.zero_grad()
		# 计算成对排序BPR损失
		prediction_i, prediction_j = model(user, item_i, item_j)
		loss = - (prediction_i - prediction_j).sigmoid().log().sum()
		# print("loss:", loss)
		total_loss += loss/len(user)
		loss.backward()
		optimizer.step()
		count += 1
	avg_loss = total_loss/count
	# 5. 模型测试
	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("Train avg_Loss: {:.3f}".format(avg_loss))
	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
	# 如果当前批次的hr指标为历史最佳，则更新最佳HR并保存模型
	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model, '{}BPR.pt'.format(config.model_path))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
