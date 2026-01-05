import torch
from pathlib import Path


# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Training Hyperparameters
# =========================
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50

# Temporal consistency weights
LAMBDA_TEMPORAL = 0.5
LAMBDA_CONF = 1.0


# =========================
# Dataset Paths
# =========================
# 根目录（根据你自己的工程路径改）
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_IMG_DIR = "/home/wangzhe/2026/IROS/Dataset/Img/apple1"
TRAIN_ANN_PATH = "/home/wangzhe/2026/IROS/Dataset/Label/apple1"

VAL_IMG_DIR = "/home/wangzhe/2026/IROS/Dataset/Img/apple2"
VAL_ANN_PATH = "/home/wangzhe/2026/IROS/Dataset/Label/apple2"


# =========================
# Checkpoint & Output
# =========================
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# 训练得到的模型权重
MODEL_SAVE_PATH = CHECKPOINT_DIR / "pmd_tsd_box.pth"

# 如果你想保存多个 epoch
MODEL_SAVE_TEMPLATE = CHECKPOINT_DIR / "pmd_tsd_box_epoch_{epoch}.pth"


# =========================
# Input / Model Settings
# =========================
# 输入尺寸（如果你统一 resize）
INPUT_WIDTH = 512
INPUT_HEIGHT = 512

# bbox 格式说明（仅用于注释 / 统一约定）
BBOX_FORMAT = "cxcywh_norm"   # center x, center y, width, height ∈ [0,1]


# =========================
# Synthetic Temporal Settings
# =========================
# synthetic temporal 仿射范围
MAX_TRANSLATION = 0.3    # 相对图像尺寸比例
MAX_SCALE = 0.3          # ±30%
MAX_ROTATION = 15        # degrees

# 裁剪 / 出画概率
CROP_PROB = 0.3
OCCLUSION_PROB = 0.3


# =========================
# Debug / Log
# =========================
PRINT_FREQ = 10          # 每多少个 batch 打印一次
SAVE_EVERY_EPOCH = False # 是否每个 epoch 都保存模型
