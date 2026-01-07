# Visual.py
# -*- coding: utf-8 -*-
"""
Save visualization images with:
  - Ground-truth boxes (GREEN)
  - Predicted boxes (RED)

Assumptions:
  batch["img"]   : Tensor [B,3,S,S] in [0,1], RGB order
  batch["boxes"] : list of Tensor[Ni,4] in xyxy pixel coords (same resized/letterboxed space as img)
  model.decode(preds, ...) returns list len B:
      {"boxes": Tensor[N,4], "scores": Tensor[N]}
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch


# -------------------------
# NMS (pure torch)
# -------------------------
@torch.no_grad()
def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float = 0.6):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thres]

    return torch.tensor(keep, device=boxes.device)


def _to_uint8_rgb(img_3chw: torch.Tensor) -> np.ndarray:
    """
    img_3chw: [3,H,W] float in [0,1]
    return: [H,W,3] uint8 RGB
    """
    img = img_3chw.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _draw_xyxy(
    img_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    scores: np.ndarray | None = None,
):
    """
    Draw boxes on RGB image using OpenCV (expects BGR color).
    boxes_xyxy: [N,4]
    """
    out = img_rgb.copy()
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, b.tolist() if hasattr(b, "tolist") else b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness)
        if scores is not None:
            s = float(scores[i])
            cv2.putText(
                out, f"{s:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2
            )
    return out


@torch.no_grad()
def visualize_and_save(
    model,
    dataloader,
    device,
    save_dir: str | Path,
    epoch: int | None = None,
    max_images: int = 5,
    conf_thres: float = 0.25,
    iou_thres: float = 0.6,
    topk: int = 300,
    img_size: int | None = None,
):
    """
    Save images with GT (green) and Pred (red).

    IMPORTANT:
      - Make sure GT boxes are in the SAME coordinate space as dataloader img.
      - If your dataset returns GT in original image coords, you must convert them
        to resized/letterboxed coords before drawing here.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for batch in dataloader:
        imgs = batch["img"].to(device)          # [B,3,S,S]
        gt_boxes_list = batch["boxes"]          # list of [Ni,4]
        metas = batch.get("meta", [None] * imgs.shape[0])

        preds = model(imgs)
        decoded = model.decode(
            preds,
            conf_thres=conf_thres,
            topk=topk,
            img_size=img_size,
        )  # list len B

        B = imgs.shape[0]
        for b in range(B):
            if count >= max_images:
                return

            img_rgb = _to_uint8_rgb(imgs[b])  # [H,W,3] RGB

            # -------- GT (GREEN) --------
            gt = gt_boxes_list[b]
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
            else:
                gt = np.asarray(gt)

            # -------- Pred (RED) --------
            pb = decoded[b]["boxes"]
            ps = decoded[b]["scores"]

            keep = nms_xyxy(pb, ps, iou_thres=iou_thres)
            pb = pb[keep].detach().cpu().numpy()
            ps = ps[keep].detach().cpu().numpy()

            # draw: GT green, Pred red
            vis = img_rgb
            if gt.shape[0] > 0:
                vis = _draw_xyxy(vis, gt, color_bgr=(0, 255, 0), thickness=2, scores=None)  # green
            if pb.shape[0] > 0:
                vis = _draw_xyxy(vis, pb, color_bgr=(0, 0, 255), thickness=2, scores=ps)    # red

            # add text
            cv2.putText(
                vis,
                f"GT:{gt.shape[0]}  Pred:{pb.shape[0]}  conf>{conf_thres:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # save (opencv expects BGR)
            if epoch is None:
                name = f"vis_{count:03d}.png"
            else:
                name = f"epoch_{epoch:03d}_{count:03d}.png"

            out_path = save_dir / name
            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"[VIS] saved {out_path} | GT={gt.shape[0]} Pred={pb.shape[0]}")

            count += 1
