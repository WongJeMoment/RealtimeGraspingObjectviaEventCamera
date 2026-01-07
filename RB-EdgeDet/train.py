# train.py
import os
from tqdm import tqdm

import torch
import torch.optim as optim

import config
from DetectionNetwork.data import DataConfig, build_dataloader
from DetectionNetwork.model import build_model
from DetectionNetwork.loss import FCOSLoss
from DetectionNetwork.utils import (
    set_seed,
    get_device,
    save_checkpoint,
)

DATASET_ROOT = config.TRAIN_IMG_DIR.parents[1]   # .../Dataset
TRAIN_SPLIT = config.TRAIN_IMG_DIR.name           # apple1
VAL_SPLIT = config.VAL_IMG_DIR.name

# -------------------------
# Train one epoch
# -------------------------
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", ncols=120)
    for i, (x5, targets) in enumerate(pbar):
        x5 = x5.to(device, non_blocking=True)

        for t in targets:
            t["boxes_xyxy"] = t["boxes_xyxy"].to(device)
            t["labels"] = t["labels"].to(device)

        outputs = model(x5)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % config.PRINT_FREQ == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_dict['loss_cls']:.3f}",
                "reg": f"{loss_dict['loss_reg']:.3f}",
                "ctr": f"{loss_dict['loss_ctr']:.3f}",
            })

    return total_loss / max(1, len(dataloader))


# -------------------------
# Validation
# -------------------------
@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch,
):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}", ncols=120)
    for x5, targets in pbar:
        x5 = x5.to(device, non_blocking=True)

        for t in targets:
            t["boxes_xyxy"] = t["boxes_xyxy"].to(device)
            t["labels"] = t["labels"].to(device)

        outputs = model(x5)
        loss = criterion(outputs, targets)["loss"]

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))


# -------------------------
# Main
# -------------------------
def main():
    # reproducibility
    set_seed(42)
    device = get_device()

    # -------------------------
    # Data
    # -------------------------
    train_data_cfg = DataConfig(
        root=DATASET_ROOT,
        split=TRAIN_SPLIT,
        img_size=config.INPUT_WIDTH,  # 或 (W,H) 看你 data.py
        num_classes=config.NUM_CLASSES,
        hflip_prob=0.5,
    )

    val_data_cfg = DataConfig(
        root=DATASET_ROOT,
        split=VAL_SPLIT,
        img_size=config.INPUT_WIDTH,
        num_classes=config.NUM_CLASSES,
        hflip_prob=0.0,
    )

    train_loader = build_dataloader(
        train_data_cfg,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    val_loader = build_dataloader(
        val_data_cfg,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # -------------------------
    # Model
    # -------------------------
    model = build_model(
        num_classes=config.NUM_CLASSES,
        fpn_out=config.FPN_OUT_CHANNELS,
        head_feat=config.HEAD_CHANNELS,
        head_convs=config.HEAD_CONVS,
        use_cepm=config.USE_CEPM,
        pretrained_backbone=config.PRETRAINED_BACKBONE,
    ).to(device)

    # -------------------------
    # Loss
    # -------------------------
    criterion = FCOSLoss(
        num_classes=config.NUM_CLASSES,
        strides=config.STRIDES,
        cls_weight=config.LOSS_CLS_WEIGHT,
        reg_weight=config.LOSS_REG_WEIGHT,
        ctr_weight=config.LOSS_CTR_WEIGHT,
    )

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    # -------------------------
    # Training loop
    # -------------------------
    best_val = 1e9
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_loss = validate(
            model, val_loader, criterion, device, epoch
        )

        print(f"[Epoch {epoch}] train: {train_loss:.4f} | val: {val_loss:.4f}")

        # save last
        save_checkpoint(
            config.LAST_MODEL_PATH,
            model,
            optimizer,
            epoch=epoch,
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                config.BEST_MODEL_PATH,
                model,
                optimizer,
                epoch=epoch,
                extra={"val_loss": val_loss},
            )

        if config.SAVE_EVERY_EPOCH:
            save_checkpoint(
                str(config.EPOCH_MODEL_TEMPLATE).format(epoch=epoch),
                model,
                optimizer,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
