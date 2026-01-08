# train.py
# -*- coding: utf-8 -*-
"""
Training entry for CCGODetector + CCGO (Innovation A).

Assumptions:
- Dataset returns:
    sample["img"]   : FloatTensor [3, S, S]
    sample["C"]     : FloatTensor [1, S, S]
    sample["boxes"] : FloatTensor [N, 4] (xyxy pixel on resized image)
- Model returns preds = [P3, P4, P5], each [B, 5, H, W]
- Loss: DetectionLoss(preds, C, boxes_list)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    set_global_seed,
    TRAIN_IMG_DIR, TRAIN_LABEL_DIR,
    VAL_IMG_DIR, VAL_LABEL_DIR,
    INPUT_SIZE, STRIDES,
    BATCH_SIZE, EPOCHS, LR,
    MODEL_SAVE_PATH, MODEL_SAVE_TEMPLATE,
    MODEL_SAVE_LATEST, MODEL_SAVE_BEST
)

from DetectionNetwork.dataset import DetectionCCGODataset, DatasetConfig
from DetectionNetwork.model import CCGODetector, DetectorConfig
from DetectionNetwork.loss import DetectionLoss, LossConfig
from Visual import visualize_and_save
from DetectionNetwork.augment import TrainAugment, AugmentConfig


# -------------------------
# Collate: variable number of boxes
# -------------------------
def collate_fn(batch: List[Dict]):
    imgs = torch.stack([b["img"] for b in batch], dim=0)  # [B,3,S,S]
    Cs = torch.stack([b["C"] for b in batch], dim=0)      # [B,1,S,S]
    boxes = [b["boxes"] for b in batch]                   # list of [Ni,4]
    metas = [b["meta"] for b in batch]
    return {"img": imgs, "C": Cs, "boxes": boxes, "meta": metas}


@torch.no_grad()
def evaluate_one_epoch(model, loss_fn, loader, device):
    model.eval()
    total = 0.0
    n = 0
    agg = {"loss_total": 0.0, "loss_box": 0.0, "loss_obj": 0.0, "loss_ccgo": 0.0}

    for batch in loader:
        img = batch["img"].to(device)
        C = batch["C"].to(device)
        boxes = batch["boxes"]

        preds = model(img)
        loss, logs = loss_fn(preds, C, boxes)

        bs = img.shape[0]
        total += float(loss.detach().cpu().item()) * bs
        n += bs
        for k in agg:
            agg[k] += float(logs.get(k, 0.0)) * bs

    for k in agg:
        agg[k] /= max(1, n)
    return agg


def main():
    # 0) seeds
    set_global_seed(42)

    device = torch.device(DEVICE)
    print("Device:", device)

    # 1) dataset
    train_cfg = DatasetConfig(
        images_dir=str(TRAIN_IMG_DIR),
        labels_dir=str(TRAIN_LABEL_DIR),
        img_size=INPUT_SIZE,
    )
    val_cfg = DatasetConfig(
        images_dir=str(VAL_IMG_DIR),
        labels_dir=str(VAL_LABEL_DIR),
        img_size=INPUT_SIZE,
    )
    augment = TrainAugment(AugmentConfig(img_size=INPUT_SIZE))
    train_ds = DetectionCCGODataset(train_cfg, augment=augment)
    val_ds = DetectionCCGODataset(val_cfg, augment=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # 2) model
    model = CCGODetector(
        DetectorConfig(
            in_channels=3,         # 如果你要 RGB+C 输入，改为 4
            base=32,
            fpn_channels=128,
            head_channels=128,
            strides=STRIDES,
        )
    ).to(device)

    # 3) loss
    loss_fn = DetectionLoss(
        LossConfig(
            img_size=INPUT_SIZE,
            strides=STRIDES,
            lambda_box=1.0,
            lambda_obj=1.0,
            lambda_ccgo=0.5,   # 创新点A权重：建议从 0.2~0.8 之间试
            center_radius=2.5,
        )
    ).to(device)

    # 4) optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 5) train loop
    best_val = float("inf")
    Path(MODEL_SAVE_LATEST).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running = 0.0
        count = 0
        for it, batch in enumerate(train_loader, start=1):
            img = batch["img"].to(device, non_blocking=True)
            C = batch["C"].to(device, non_blocking=True)
            boxes = batch["boxes"]

            preds = model(img)
            loss, logs = loss_fn(preds, C, boxes)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            bs = img.shape[0]
            running += float(loss.detach().cpu().item()) * bs
            count += bs

            if it % 20 == 0:
                print(
                    f"[Epoch {epoch:03d}/{EPOCHS:03d}] "
                    f"Iter {it:04d} | "
                    f"loss={logs['loss_total']:.4f} "
                    f"(box={logs['loss_box']:.4f}, obj={logs['loss_obj']:.4f}, ccgo={logs['loss_ccgo']:.4f})"
                )

        train_loss = running / max(1, count)

        # 6) validate
        val_logs = evaluate_one_epoch(model, loss_fn, val_loader, device)
        val_loss = val_logs["loss_total"]
        visualize_and_save(
            model=model,
            dataloader=val_loader,
            device=device,
            save_dir="bbox_vis",
            epoch=epoch,
            max_images=20,
            conf_thres=0.3,
            iou_thres=0.6,
        )

        print(
            f"\nEpoch {epoch:03d} done | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"(box={val_logs['loss_box']:.4f}, obj={val_logs['loss_obj']:.4f}, ccgo={val_logs['loss_ccgo']:.4f})\n"
        )

        # 7) save (only latest + best, .pth)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "cfg": {
                "INPUT_SIZE": INPUT_SIZE,
                "STRIDES": STRIDES,
                "LR": LR,
                "BATCH_SIZE": BATCH_SIZE,
            },
        }

        # ---- save latest (always overwrite) ----
        torch.save(ckpt, MODEL_SAVE_LATEST)

        # ---- save best (only if improved) ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, MODEL_SAVE_BEST)
            print(
                f"✅ Best model updated: {MODEL_SAVE_BEST} "
                f"(val_loss={best_val:.6f})"
            )

    print("Training finished.")


if __name__ == "__main__":
    main()
