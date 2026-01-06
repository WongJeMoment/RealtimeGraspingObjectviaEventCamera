import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from TrackingNetwork.model import PMD_TSD_Box
from TrackingNetwork.LossFunction import temporal_loss
from Dataset import SingleFrameDataset
import config as cfg

# 👇 引入可视化工具
from vis_utils import save_vis_images

# =========================
# 基本配置
# =========================
DEVICE = cfg.DEVICE
BATCH_SIZE = cfg.BATCH_SIZE
LR = cfg.LR
EPOCHS = cfg.EPOCHS

LAMBDA_TEMPORAL = cfg.LAMBDA_TEMPORAL
LAMBDA_CONF = cfg.LAMBDA_CONF

VIS_ROOT = "vis"
SAVE_VIS_EVERY = 100
MAX_VIS_IMAGES = 4


# =========================
# synthetic temporal
# =========================
def synthetic_temporal_pair(img, bbox):
    img_tm1 = img.clone()
    img_t = img.clone()
    bbox_tm1 = bbox.clone()
    bbox_t = bbox.clone()
    return img_tm1, img_t, bbox_tm1, bbox_t


def bbox_l1_loss(pred, gt):
    return torch.abs(pred - gt).mean()


# =========================
# 训练主函数
# =========================
def train():
    # 1) 构造 image path list
    img_dir = cfg.TRAIN_IMG_DIR
    img_paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    dataset = SingleFrameDataset(
        data_list=img_paths,
        label_dir=cfg.TRAIN_ANN_PATH,
        img_size=(cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH),
        augment=True
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    model = PMD_TSD_Box().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    os.makedirs(VIS_ROOT, exist_ok=True)
    global_step = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in loader:
            img = batch["img"].to(DEVICE)
            bbox = batch["bbox"].to(DEVICE)

            img_tm1, img_t, box_tm1, box_t_gt = synthetic_temporal_pair(img, bbox)

            pred_box_t, conf = model(img_t, img_tm1, box_tm1.detach())

            loss_bbox = bbox_l1_loss(pred_box_t, box_t_gt)
            loss_conf = nn.functional.binary_cross_entropy(
                conf, torch.ones_like(conf)
            )

            visibility = torch.ones(pred_box_t.size(0), device=DEVICE)
            loss_temporal = temporal_loss(
                pred_box_t,
                box_tm1,
                box_tm1,
                visibility
            )

            loss = (
                loss_bbox
                + LAMBDA_CONF * loss_conf
                + LAMBDA_TEMPORAL * loss_temporal
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ===== 可视化保存 =====
            if global_step % SAVE_VIS_EVERY == 0:
                save_dir = os.path.join(VIS_ROOT, f"epoch_{epoch:03d}")
                save_vis_images(
                    img_batch=img_t,
                    gt_batch=box_t_gt,
                    pred_batch=pred_box_t,
                    save_dir=save_dir,
                    step=global_step,
                    max_show=MAX_VIS_IMAGES
                )

            global_step += 1

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Loss: {total_loss / len(loader):.4f}"
        )

    torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
