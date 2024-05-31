import json
import argparse

"""
本文件，用于解析命令行参数
"""

all_datasets = ["lastfm", "gowalla", "yelp2018", "amazon-book", "citeulike", "movielens", "amazon-beauty", "amazon-cds", "amazon-electro", "amazon-movies"]
all_models = ["mf", "lgn", "base-a-lgn", "finer-a-lgn", "sdp-a-lgn", "w-sdp-a-lgn", "appnp"]


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="BPR_loss训练过程的批量大小")
    parser.add_argument("--recdim", type=int, default=64,
                        help="LightGCN 的嵌入大小")
    parser.add_argument("--layer", type=int, default=3,
                        help="LightGCN的层数")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--decay", type=float, default=1e-4,
                        help="L2正则化的权重衰减")
    parser.add_argument("--dropout", type=int, default=0,
                        help="是否使用dropout")
    parser.add_argument("--keepprob", type=float, default=0.6,
                        help="dropout的保持概率")
    parser.add_argument("--a_fold", type=int, default=100,
                        help="分割大的邻接矩阵的折数")
    parser.add_argument("--testbatch", type=int, default=100,
                        help="用于测试的用户批量大小")
    parser.add_argument("--dataset", type=str, default="gowalla",
                        help=f"available datasets: {str(all_datasets)}")
    parser.add_argument("--path", type=str, default="./checkpoints",
                        help="保存权重的路径")
    parser.add_argument("--topks", nargs="?", default="[1, 2, 3, 5, 10, 20]",
                        help="@k test list")
    parser.add_argument("--comment", type=str, default="lgn")  # 注释，默认为 "lgn"
    parser.add_argument("--load", type=int, default=0)  # 是否加载模型
    parser.add_argument("--epochs", type=int, default=600)  # 训练时的迭代次数
    parser.add_argument("--multicore", type=int, default=0,
                        help="是否在测试时使用多核")
    parser.add_argument("--pretrain", type=int, default=0,
                        help="是否使用预训练权重")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")  # 随机种子
    parser.add_argument("--model", type=str, default="lgn",
                        help=f"rec-model, support {str(all_models)}")  # 推荐模型名称
    parser.add_argument("--single", action="store_true", default=False,
                        help="whether to use single LightGCN to test")
    parser.add_argument("--save_embs", action="store_true", default=False,
                        help=" 是否保存嵌入矩阵")
    parser.add_argument("--l1", action="store_true", default=False,
                        help="是否使用L1范数进行邻接矩阵规范化")
    parser.add_argument("--side_norm", type=str, default="both",
                        help="available norms: [l, r, both]")
    parser.add_argument("--save_model_by", type=str, default="ndcg",
                        help="available metrics: [ndcg, recall, precision]")

    # Optimization
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="optimizer to use for training")  # 优化器
    parser.add_argument("--scheduler", type=str, default="step_lr",
                        help="scheduler to use for adjusting learning rate")  # 学习率调度器
    parser.add_argument("--scheduler_params", type=json.loads, default={},
                        help="more params for the scheduler in JSON format")  # 指定调度器额外参数，以JSON格式提供

    # 设置注意力机制中注意力投影的维度，默认为2
    parser.add_argument("--attention_dim", type=int, default=8,
                        help="Number of dims for the attention projections")

    # APPNP
    parser.add_argument('--num_walks', type=int, default=10,
                        help='Number of random walk steps for APPNP')  # 随机游走步数
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='The teleportation parameter for APPNP')  # 传输参数

    return parser.parse_args()
