import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
# 读取CSV文件
csv_file_path = 'user_bins.csv'
df = pd.read_csv(csv_file_path)

# 统计Value列中不同取值的数量
value_counts = df['Value'].value_counts()

# 绘制柱状图
plt.bar(value_counts.index, value_counts.values)
plt.xlabel('Bin')
plt.ylabel('Count')
plt.title('不同分箱中的用户数量（gowalla数据集）')
plt.xticks(list(value_counts.index), list(map(int, value_counts.index)))
plt.show()

# # 数据
# data = {
#     'precision': [0.3458, 0.311, 0.5244, 0.4276, 0.5232, 0.391, 0.5375, 0.4064, 0.3982, 0.3176,
#                   0.2657, 0.2135, 0.1691, 0.1232, 0.0913, 0.05184, 0.04532, 0.02107, 0.0107, 0.003512],
#     'recall': [2.807, 2.075, 3.058, 2.037, 1.981, 1.213, 1.349, 0.8038, 0.6282, 0.3958,
#                0.2603, 0.1641, 0.1023, 0.05944, 0.03424, 0.01551, 0.01056, 0.003963, 0.00151, 0.0003948],
#     'ndcg': [0.141, 0.1258, 0.1255, 0.1611, 0.1682, 0.1491, 0.1727, 0.1527, 0.1408, 0.1433,
#              0.1248, 0.1109, 0.1212, 0.08053, 0.07487, 0.04271, 0.03867, 0.02253, 0.009873, 0.004028]
# }
# df = pd.DataFrame(data)
#
# # 归一化每一列
# df['precision'] = (df['precision'] - df['precision'].min()) / (df['precision'].max() - df['precision'].min())
# df['recall'] = (df['recall'] - df['recall'].min()) / (df['recall'].max() - df['recall'].min())
# df['ndcg'] = (df['ndcg'] - df['ndcg'].min()) / (df['ndcg'].max() - df['ndcg'].min())
#
# # 转换为numpy数组
# precision_np = df['precision'].values
# recall_np = df['recall'].values
# ndcg_np = df['ndcg'].values
# index_np = df.index.values
#
# # 设置图形大小
# plt.figure(figsize=(10, 6))
#
# # 绘制折线图，使用numpy数组
# plt.plot(index_np, precision_np, label='Precision', marker='o')  # Precision 数据
# plt.plot(index_np, recall_np, label='Recall', marker='o')        # Recall 数据
# plt.plot(index_np, ndcg_np, label='NDCG', marker='o')            # NDCG 数据
#
# # 添加标题和标签
# plt.title('不同分箱用户组的推荐性能表现（gowalla数据集）')
# plt.xlabel('Bin')
# plt.ylabel('Mapping Value')
# plt.xticks(index_np)
#
# # 添加图例
# plt.legend()
#
# # 显示图表
# plt.show()
