import os
import torch
import sys
import multiprocessing

from parse import parse_args
from parse import all_datasets
from parse import all_models
from os.path import join
from warnings import simplefilter

"""
设置和配置实验的环境和参数，以及加载数据集和模型。
"""

# 设置环境变量KMP_DUPLICATE_LIB_OK，解决库重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 获取用户在命令行中输入的参数
args = parse_args()
# 设置WandB实验追踪的项目和实体名称
WANDB_PROJECT = "recsys"
WANDB_ENTITY = "msc-ai"
# 获取根路径，即当前文件的上一级目录
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
# 设置代码、数据和其他相关文件的路径
CODE_PATH = join(ROOT_PATH, "code")
DATA_PATH = join(ROOT_PATH, "data")
BOARD_PATH = join(CODE_PATH, "runs")
EMBS_PATH = join(CODE_PATH, "embs")
FILE_PATH = join(CODE_PATH, "checkpoints")
# 将源代码文件夹的路径添加到Python搜索路径中，以便导入自定义模块
sys.path.append(join(CODE_PATH, "sources"))
for path_name in [FILE_PATH, EMBS_PATH]:
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)

# 初始化配置字典，用于存储实验的参数
config = {}
# 对各种实验参数进行设置
config["batch_size"] = args.batch_size
config["latent_dim_rec"] = args.recdim
config["lightGCN_n_layers"] = args.layer
config["dropout"] = args.dropout
config["keep_prob"] = args.keepprob
config["adj_matrix_folds"] = args.a_fold
config["test_u_batch_size"] = args.testbatch
config["multicore"] = args.multicore
config["lr"] = args.lr
config["decay"] = args.decay
config["pretrain"] = args.pretrain
config["A_split"] = False
config["bigdata"] = False
config["seed"] = args.seed
config["topks"] = eval(args.topks)
config["single"] = args.single
config["l1"] = args.l1
config["side_norm"] = args.side_norm
config["embs_path"] = EMBS_PATH
config["save_embs"] = args.save_embs
config["dataset"] = args.dataset
config["model"] = args.model
config["save_model_by"] = args.save_model_by
# Attention
if "attention_dim" in args and config["model"] == "w-sdp-a-lgn":
    config["attention_dim"] = args.attention_dim
# APPNP
if 'num_walks' in args and 'alpha' in args and config['model'] == 'appnp':
    config["num_walks"] = args.num_walks
    config["alpha"] = args.alpha
# 检测是否有可用的 GPU，并将模型移动到 GPU 或 CPU 上
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed
# 获取数据集和模型的名称
dataset = config["dataset"]
model_name = config["model"]
# 如果数据集或模型名称不在预定义的列表中，则抛出 NotImplementedError
if config["dataset"] not in all_datasets:
    raise NotImplementedError(
        f"Haven't supported {config['dataset']} yet!, try {all_datasets}")
if config["model"] not in all_models:
    raise NotImplementedError(
        f"Haven't supported {config['model']} yet!, try {all_models}")
# 设置训练的 epoch 数、是否加载模型、模型路径等参数
TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
comment = args.comment
num_bins = 20

# 设置忽略未来警告
simplefilter(action="ignore", category=FutureWarning)
