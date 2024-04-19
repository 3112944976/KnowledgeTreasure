import pandas as pd
import matplotlib.pyplot as plt


# 可视化结果
def visualize_multiple_test_results(test_files):
    plt.figure(figsize=(10, 6))
    line_styles = ['-', '--', '-.', ':']  # 定义不同的线条样式
    max_epoch = 0  # 初始化最大的 Epoch 值
    for i, test_file in enumerate(test_files):
        test_data = pd.read_csv(test_file)
        max_epoch = max(max_epoch, test_data['Epoch'].max())  # 更新最大的 Epoch 值
        line_style = line_styles[i % len(line_styles)]  # 循环使用线条样式
        plt.plot(test_data['Epoch'].values, test_data['LOSS'].values, label='GRU_NeuMF: LOSS', linestyle=line_style)
        plt.plot(test_data['Epoch'].values, test_data['NDCG'].values, label='GRU_NeuMF: NDCG', linestyle=line_style)
        plt.plot(test_data['Epoch'].values, test_data['HR'].values, label='GRU_NeuMF: HR', linestyle=line_style)
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('TestSet-Based Model Performance')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(int(max_epoch) + 1))  # X轴刻度只显示整数
    plt.show()


if __name__ == "__main__":
    test_files = ['./result/test_resultsGRU128.csv']
    visualize_multiple_test_results(test_files)