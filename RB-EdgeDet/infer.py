# Visual.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# =========================
# ✅ 一键运行配置区（只看这里）
# =========================
IMAGES_DIR = "/home/wangzhe/2026/IROS/Dataset/Img/apple2"
LABELS_DIR = "/home/wangzhe/2026/IROS/Dataset/Label/apple2"

CKPT_PATH = "/home/wangzhe/2026/IROS/RB-EdgeDet/checkpoints/ccgo_detector_best.pth"

IMG_SIZE = 640
BATCH_SIZE = 2
MAX_IMAGES = 30

CONF_THRES = 0.25
IOU_THRES = 0.6
TOPK = 300

SAVE_DIR = "./vis_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Imports from your project
# =========================
from DetectionNetwork.dataset import DetectionCCGODataset, DatasetConfig
from DetectionNetwork.model import CCGODetector, DetectorConfig


# -------------------------
# NMS
# -------------------------
@torch.no_grad()
def nms_xyxy(boxes, scores, iou_thres=0.6):
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


def to_uint8_rgb(img_3chw: torch.Tensor) -> np.ndarray:
    img = img_3chw.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def draw_boxes(img_rgb, boxes, color_bgr, scores=None):
    out = img_rgb.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        if scores is not None:
            cv2.putText(
                out, f"{scores[i]:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2
            )
    return out


# =========================
# ✅ 一键运行主函数
# =========================
@torch.no_grad()
def main():
    print("🚀 Visualizing CCGO Detector")
    print("Device:", DEVICE)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------- Dataset --------
    ds_cfg = DatasetConfig(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        img_size=IMG_SIZE,
    )
    dataset = DetectionCCGODataset(ds_cfg, augment=None)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: {
            "img": torch.stack([x["img"] for x in b], 0),
            "boxes": [x["boxes"] for x in b],
        },
    )

    # -------- Model --------
    model = CCGODetector(
        DetectorConfig(
            in_channels=3,
            base=32,
            fpn_channels=128,
            head_channels=128,
        )
    ).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    print("Loaded checkpoint:", CKPT_PATH)

    # -------- Visualize --------
    count = 0
    for batch in loader:
        imgs = batch["img"].to(DEVICE)
        gts = batch["boxes"]

        preds = model(imgs)
        decoded = model.decode(
            preds,
            conf_thres=CONF_THRES,
            topk=TOPK,
            img_size=IMG_SIZE,
        )

        for i in range(imgs.shape[0]):
            if count >= MAX_IMAGES:
                print("✅ Done.")
                return

            img = to_uint8_rgb(imgs[i])
            gt = gts[i].cpu().numpy() if isinstance(gts[i], torch.Tensor) else gts[i]

            pb = decoded[i]["boxes"]
            ps = decoded[i]["scores"]

            keep = nms_xyxy(pb, ps, IOU_THRES)
            pb = pb[keep]
            ps = ps[keep]

            # ✅ 只保留置信度最高的框（Top-1）
            if ps.numel() > 0:
                top1 = torch.argmax(ps)
                pb = pb[top1:top1 + 1]
                ps = ps[top1:top1 + 1]

            pb = pb.cpu().numpy()
            ps = ps.cpu().numpy()

            vis = img
            if gt.shape[0] > 0:
                vis = draw_boxes(vis, gt, (0, 255, 0))      # GT green
            if pb.shape[0] > 0:
                vis = draw_boxes(vis, pb, (0, 0, 255), ps)  # Pred red

            cv2.putText(
                vis,
                f"GT:{gt.shape[0]}  Pred:{pb.shape[0]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )

            out_path = save_dir / f"vis_{count:04d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print("Saved:", out_path)

            count += 1

    print("✅ Done.")


if __name__ == "__main__":
    main()
