import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data

import config


def load_all(file_path1, file_path2, test_neg_num):
	"""处理基于隐式反馈的数据集amazon-electro"""
	# 1. 加载训练数据
	print("加载训练数据...")
	data = []
	with open(file_path1, 'r') as file:
		for line in file:
			line = list(map(int, line.strip().split()))
			user_id = line[0]
			for item_id in line[1:]:
				data.append([user_id, item_id])
	train_data = pd.DataFrame(data, columns=['user', 'item'])
	# 2. 获取用户和物品数
	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1
	# 3. 获取稀疏评分矩阵
	train_data = train_data.values.tolist()
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0
	# 4. 加载测试数据
	print("加载测试数据...")
	test_data = []
	with open(file_path2, 'r') as file:
		for line in file:
			line_data = list(map(int, line.strip().split()))
			# 存储一个行用户u的正例测试样本
			user_id = line_data[0]
			try:
				test_data.append([user_id, line_data[1]])
			except IndexError:
				print(f"测试集中用户{user_id}不存在正例物品")
				continue
			# 追加存储test_neg_num个行用户u的负例测试样本
			for _ in range(test_neg_num):
				j = np.random.randint(item_num)
				while (user_id, j) in train_mat or j in line_data[2:]:
					j = np.random.randint(item_num)
				test_data.append([user_id, j])
	return train_data, test_data, user_num, item_num, train_mat


class BPRData(data.Dataset):
	"""
		features: [[user_id, item_id], ...]
		num_ng：负样本数，默认为0
		is_training：是否处于训练模式的布尔值
	"""
	def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
	# 1. 负样本采样（仅训练时需要）
	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'
		print("采样训练样本，开始迭代学习...")
		self.features_fill = []
		# 遍历features中的每个样本[user_id, item_id]
		for x in self.features:
			u, i = x[0], x[1]
			# 随机采样num_ng个负样本
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				# 确保随机采样到的负样本不在train_mat中，以避免正样本
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				# 保存三元组样本[u, i, j]
				self.features_fill.append([u, i, j])
	# 2. 返回样本数
	def __len__(self):
		return self.num_ng * len(self.features) if self.is_training else len(self.features)
	# 3. 按索引获取样本(user_id, item_i, item_j)：测试时item_i=item_j，训练时item_j为负样本
	def __getitem__(self, idx):
		features = self.features_fill if self.is_training else self.features
		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if self.is_training else features[idx][1]
		return user, item_i, item_j 
		