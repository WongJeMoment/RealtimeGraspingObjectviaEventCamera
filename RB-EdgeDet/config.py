# src/utils/config.py
# -*- coding: utf-8 -*-
"""
Unified configuration & utility module.

Includes:
  - YAML config loading
  - Random seed fixing (reproducibility)
  - Bounding box utilities (format convert, IoU, clipping)
  - Dataset paths (train / val / test)

Safe to import anywhere.
"""

from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import os
import random
import yaml
import numpy as np
import torch


# ============================================================
# 0. PROJECT ROOT
# ============================================================
# 当前工程根目录（src/utils/config.py -> 回到工程根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 1. YAML CONFIG
# ============================================================
def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ============================================================
# 2. DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 3. REPRODUCIBILITY (SEED)
# ============================================================
def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 4. DATASET PATHS
# ============================================================
# 你所有数据的根目录（可以统一管理）
DATA_ROOT = Path("/home/wangzhe/2026/IROS/Dataset")

# -------- Train --------
TRAIN_IMG_DIR = DATA_ROOT / "Img" / "apple1"
TRAIN_LABEL_DIR = DATA_ROOT / "Label" / "apple1"

# -------- Validation --------
VAL_IMG_DIR = DATA_ROOT / "Img" / "apple2"
VAL_LABEL_DIR = DATA_ROOT / "Label" / "apple2"

# -------- Test (可选，没有就留着) --------
TEST_IMG_DIR = DATA_ROOT / "Img" / "apple_test"
TEST_LABEL_DIR = DATA_ROOT / "Label" / "apple_test"


# ============================================================
# 5. TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50

# Loss weights
LAMBDA_CCGO = 0.5
LAMBDA_OBJ = 1.0
LAMBDA_BOX = 1.0


# ============================================================
# 6. INPUT / MODEL SETTINGS
# ============================================================
INPUT_SIZE = 640          # 统一 resize 到正方形
STRIDES = (8, 16, 32)

# bbox 格式说明（用于约定）
BBOX_FORMAT = "xyxy_pixel"   # after resize


# ============================================================
# 7. CHECKPOINT & OUTPUT
# ============================================================
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_SAVE_PATH = CHECKPOINT_DIR / "ccgo_detector.pth"
MODEL_SAVE_TEMPLATE = CHECKPOINT_DIR / "ccgo_detector_epoch_{epoch}.pth"


# ============================================================
# 8. DATA AUGMENT (如果你后面要用)
# ============================================================
MAX_TRANSLATION = 0.3    # relative to image size
MAX_SCALE = 0.3
MAX_ROTATION = 15        # degrees

CROP_PROB = 0.3
OCCLUSION_PROB = 0.3


# ============================================================
# 9. BOX UTILITIES
# ============================================================
def clip_boxes_xyxy(
    boxes: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Clip boxes to image boundaries.

    boxes: Tensor[..., 4] in xyxy
    """
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0].clamp(0, width - 1)
    boxes[..., 2] = boxes[..., 2].clamp(0, width - 1)
    boxes[..., 1] = boxes[..., 1].clamp(0, height - 1)
    boxes[..., 3] = boxes[..., 3].clamp(0, height - 1)
    return boxes


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU between two sets of boxes (paired).
    """
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    inter_x1 = torch.max(ax1, bx1)
    inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2)
    inter_y2 = torch.min(ay2, by2)

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)

    union = area_a + area_b - inter + eps
    return inter / union


# ============================================================
# 10. PRETTY PRINT
# ============================================================
def print_config(cfg: Dict[str, Any], indent: int = 0):
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(v, indent + 2)
        else:
            print(" " * indent + f"{k}: {v}")
