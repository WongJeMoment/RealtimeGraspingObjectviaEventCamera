# vis_utils.py
import os
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes


def _clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0.0, min(float(x1), float(W - 1)))
    y1 = max(0.0, min(float(y1), float(H - 1)))
    x2 = max(0.0, min(float(x2), float(W - 1)))
    y2 = max(0.0, min(float(y2), float(H - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def cxcywh_norm_to_xyxy_px(box, W, H):
    """
    box: Tensor [4] (cx,cy,w,h), normalized [0,1]
    return: [x1,y1,x2,y2] in pixels
    """
    cx, cy, w, h = box.tolist()
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return _clamp_xyxy(x1, y1, x2, y2, W, H)


@torch.no_grad()
def save_vis_images(
    img_batch,
    gt_batch,
    pred_batch,
    save_dir,
    step,
    max_show=4,
):
    """
    保存 GT bbox + Pred bbox 的可视化图片

    Args:
        img_batch:  Tensor [B,3,H,W], float in [0,1]
        gt_batch:   Tensor [B,4], cxcywh_norm
        pred_batch: Tensor [B,4], cxcywh_norm
        save_dir:   保存目录
        step:       global step（用于命名）
        max_show:   最多保存几张
    """
    os.makedirs(save_dir, exist_ok=True)

    imgs = img_batch.detach().clamp(0, 1).cpu()
    gts = gt_batch.detach().cpu()
    preds = pred_batch.detach().cpu()

    B, _, H, W = imgs.shape
    n = min(B, max_show)

    for i in range(n):
        img_u8 = (imgs[i] * 255.0).to(torch.uint8)

        gt_xyxy = torch.tensor(
            [cxcywh_norm_to_xyxy_px(gts[i], W, H)],
            dtype=torch.float32
        )
        pred_xyxy = torch.tensor(
            [cxcywh_norm_to_xyxy_px(preds[i], W, H)],
            dtype=torch.float32
        )

        out = draw_bounding_boxes(
            img_u8,
            gt_xyxy,
            colors="green",
            width=2,
            labels=["GT"],
        )
        out = draw_bounding_boxes(
            out,
            pred_xyxy,
            colors="red",
            width=2,
            labels=["Pred"],
        )

        save_path = os.path.join(
            save_dir, f"step_{step:06d}_img{i}.png"
        )
        torchvision.utils.save_image(out.float() / 255.0, save_path)
