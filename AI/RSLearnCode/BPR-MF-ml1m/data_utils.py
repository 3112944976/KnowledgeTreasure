import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data

import config


def load_all(test_num=100):
	# 1. 加载训练数据
	train_data = pd.read_csv(
		config.train_rating, 
		sep='\t', header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
	# 获取用户和物品数
	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1
	# 将train_data转换为list形式
	train_data = train_data.values.tolist()
	# 创建稀疏评分矩阵train_mat
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	# 通过遍历训练样本，将对应位置的评分设为1.0
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	# 2. 加载测试数据
	test_data = []
	# test_negative中每行的第一个正例样本元素(u,i)和test_rating中的一致：凡是评过分的，都认为是正例物品。
	with open(config.test_negative, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			# 读取每行的第一个元素'(u,i)'中的u
			u = eval(arr[0])[0]
			# 存储该行第一个[u, i]，此为该用户的正例样本
			test_data.append([u, eval(arr[0])[1]])
			# 读取每行剩余的其它元素i
			for i in arr[1:]:
				# 存储该用户剩余的99个负例样本[u, i]
				test_data.append([u, int(i)])
			line = fd.readline()
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
				# 保存三元组样本[u, i, j]（若要改回MSE损失，这里需追加一个[u,i]和num_ng个[u,j]，并额外构造label）
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
		