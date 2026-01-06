# val.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TrackingNetwork.model import PMD_TSD_Box
from TrackingNetwork.LossFunction import temporal_loss
from Dataset import SingleFrameDataset
import config as cfg

from vis_utils import save_vis_images


# =========================
# 基本配置
# =========================
DEVICE = cfg.DEVICE
BATCH_SIZE = cfg.BATCH_SIZE

LAMBDA_TEMPORAL = cfg.LAMBDA_TEMPORAL
LAMBDA_CONF = cfg.LAMBDA_CONF

VAL_VIS_ROOT = "vis_val"
# 验证时你说想“每张都可视化”，那就每个 batch 都保存
SAVE_ALL_VAL = True
MAX_VIS_IMAGES = 999999


# =========================
# synthetic temporal（和 train 保持一致）
# =========================
def synthetic_temporal_pair(img, bbox):
    img_tm1 = img.clone()
    img_t = img.clone()
    bbox_tm1 = bbox.clone()
    bbox_t = bbox.clone()
    return img_tm1, img_t, bbox_tm1, bbox_t


def bbox_l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


@torch.no_grad()
def validate():
    # 1) 构造 val image path list
    img_dir = cfg.VAL_IMG_DIR
    img_paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    dataset = SingleFrameDataset(
        data_list=img_paths,
        label_dir=cfg.VAL_ANN_PATH,
        img_size=(cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH),
        augment=False  # 验证不做增强
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # 2) 加载模型
    model = PMD_TSD_Box().to(DEVICE)
    ckpt_path = str(cfg.MODEL_SAVE_PATH)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3) 验证循环
    os.makedirs(VAL_VIS_ROOT, exist_ok=True)

    total_loss = 0.0
    total_bbox = 0.0
    total_conf = 0.0
    total_temp = 0.0

    # 你可以把这里当成“val epoch”
    epoch = 0
    global_step = 0

    for batch_idx, batch in enumerate(loader):
        img = batch["img"].to(DEVICE)     # [B,3,H,W]
        bbox = batch["bbox"].to(DEVICE)   # [B,4]

        img_tm1, img_t, box_tm1, box_t_gt = synthetic_temporal_pair(img, bbox)

        pred_box_t, conf = model(img_t, img_tm1, box_tm1)

        loss_bbox = bbox_l1_loss(pred_box_t, box_t_gt)
        loss_conf = nn.functional.binary_cross_entropy(conf, torch.ones_like(conf))

        visibility = torch.ones(pred_box_t.size(0), device=DEVICE)
        loss_temporal = temporal_loss(
            pred_box_t,
            box_tm1,
            box_tm1,
            visibility
        )

        loss = loss_bbox + LAMBDA_CONF * loss_conf + LAMBDA_TEMPORAL * loss_temporal

        bs = img.size(0)
        total_loss += loss.item() * bs
        total_bbox += loss_bbox.item() * bs
        total_conf += loss_conf.item() * bs
        total_temp += loss_temporal.item() * bs

        # 4) 保存可视化：默认每个 batch 都保存（等价于每张都保存）
        if SAVE_ALL_VAL:
            save_dir = os.path.join(VAL_VIS_ROOT, f"epoch_{epoch:03d}")
            # 这里每次最多保存 batch_size 张，设置 max_show=B 就可以全保存
            save_vis_images(
                img_batch=img_t,
                gt_batch=box_t_gt,
                pred_batch=pred_box_t,
                save_dir=save_dir,
                step=global_step,         # 用 global_step 区分文件名
                max_show=img_t.size(0)    # ✅ 这一行保证一个 batch 全保存
            )

        global_step += 1

    n = len(dataset)
    mean_loss = total_loss / max(1, n)
    mean_bbox = total_bbox / max(1, n)
    mean_conf = total_conf / max(1, n)
    mean_temp = total_temp / max(1, n)

    print("========== Validation ==========")
    print(f"Num samples: {n}")
    print(f"Loss total : {mean_loss:.6f}")
    print(f"Loss bbox  : {mean_bbox:.6f}")
    print(f"Loss conf  : {mean_conf:.6f}")
    print(f"Loss temp  : {mean_temp:.6f}")
    print(f"Vis saved  : {os.path.abspath(VAL_VIS_ROOT)}")


if __name__ == "__main__":
    validate()
