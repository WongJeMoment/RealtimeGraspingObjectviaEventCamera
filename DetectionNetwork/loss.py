# src/losses/loss.py
# -*- coding: utf-8 -*-
"""
Unified loss for CCGODetector:
  - Center-sampling assignment (anchor-free)
  - CIoU box regression loss
  - Focal BCE objectness loss
  - CCGO: color-consistency guided objectness alignment loss
  - Total loss aggregation

Model output:
  preds = [P3, P4, P5]
  each Pi: Tensor[B, 5, Hi, Wi] with channels (l,t,r,b,obj_logit)

Dataset batch:
  batch["img"]:  [B, 3, S, S] float
  batch["C"]:    [B, 1, S, S] float in [0,1]
  batch["boxes"]: list[Tensor[Ni,4]] or Tensor with varying N (use collate to list)
                 format xyxy in pixel coords on the resized image (SxS)

Dependencies:
  torch
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Config
# -------------------------
@dataclass
class LossConfig:
    img_size: int = 640
    strides: Tuple[int, int, int] = (8, 16, 32)

    # assignment
    center_radius: float = 2.5  # in feature-map cells (like FCOS center sampling)
    # ignore area around positives (optional). set 0 to disable.
    ignore_radius: float = 0.0

    # loss weights
    lambda_box: float = 1.0
    lambda_obj: float = 1.0
    lambda_ccgo: float = 0.5

    # focal for objectness
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # ccgo per-level weights
    ccgo_level_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # stabilize ltrb regression
    ltrb_min: float = 0.0
    ltrb_max: float = 1e4


# -------------------------
# Utils: boxes & iou
# -------------------------
def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)


def _box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    a: [N,4], b: [N,4] (paired)
    returns IoU [N]
    """
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_a = _box_area_xyxy(a)
    area_b = _box_area_xyxy(b)
    union = area_a + area_b - inter + eps
    return inter / union


def ciou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    CIoU loss for paired boxes: 1 - CIoU
    pred_xyxy, tgt_xyxy: [N,4]
    """
    iou = _box_iou_xyxy(pred_xyxy, tgt_xyxy, eps=eps)

    pc = _xyxy_to_cxcywh(pred_xyxy)
    tc = _xyxy_to_cxcywh(tgt_xyxy)
    px, py, pw, ph = pc.unbind(-1)
    tx, ty, tw, th = tc.unbind(-1)

    # center distance
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    # enclosing box diagonal
    ex1 = torch.minimum(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    ey1 = torch.minimum(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    ex2 = torch.maximum(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    ey2 = torch.maximum(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    c2 = ((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2).clamp(min=eps)

    # aspect ratio term
    v = (4 / (torch.pi ** 2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return 1.0 - ciou


# -------------------------
# Focal BCE for objectness
# -------------------------
def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    logits/targets: any shape, targets in {0,1} (float ok)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    mod = (1 - p_t) ** gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * mod * bce
    else:
        loss = mod * bce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# -------------------------
# Assignment (center sampling)
# -------------------------
def _build_grid_centers(H: int, W: int, stride: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns cx, cy: [H,W] in image pixel coords
    """
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    return cx, cy


def assign_targets_center_sampling(
    boxes_xyxy: torch.Tensor,
    H: int,
    W: int,
    stride: int,
    center_radius: float,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build dense targets on one feature level.

    Args:
      boxes_xyxy: [N,4] gt boxes in image pixels (on resized image)
    Returns:
      obj_t: [1,H,W] {0,1}
      ltrb_t: [4,H,W] regression targets (in pixels / stride? we keep in "cell units" for stability)
              Here we store in "cell units" (divide by stride) to match model output scale.
    """
    obj_t = torch.zeros((1, H, W), device=device)
    ltrb_t = torch.zeros((4, H, W), device=device)

    if boxes_xyxy.numel() == 0:
        return obj_t, ltrb_t

    cx, cy = _build_grid_centers(H, W, stride, device=device)  # [H,W]

    # For each location, we will select the best-matching GT among candidates
    # using minimal GT area (common FCOS rule) to resolve overlaps.
    INF = 1e9
    best_area = torch.full((H, W), INF, device=device)
    best_ltrb = torch.zeros((4, H, W), device=device)

    for b in boxes_xyxy:
        x1, y1, x2, y2 = b
        # valid box
        if (x2 <= x1 + 1) or (y2 <= y1 + 1):
            continue

        # location inside box?
        in_box = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

        # center sampling constraint (square around gt center in feature coords)
        gx = (x1 + x2) * 0.5
        gy = (y1 + y2) * 0.5
        rad = center_radius * stride
        in_center = (cx >= gx - rad) & (cx <= gx + rad) & (cy >= gy - rad) & (cy <= gy + rad)

        pos = in_box & in_center
        if pos.sum() == 0:
            continue

        area = (x2 - x1) * (y2 - y1)
        # only update where this gt has smaller area (prefer small objects for overlaps)
        update = pos & (area < best_area)

        if update.sum() == 0:
            continue

        # l,t,r,b in pixel distances from center point to edges
        l = (cx - x1).clamp(min=0)
        t = (cy - y1).clamp(min=0)
        r = (x2 - cx).clamp(min=0)
        btm = (y2 - cy).clamp(min=0)

        # store in "cell units" (divide by stride) so magnitudes are comparable across levels
        ltrb = torch.stack([l, t, r, btm], dim=0) / float(stride)  # [4,H,W]

        best_area[update] = area
        best_ltrb[:, update] = ltrb[:, update]

    # positives are where best_area was set
    pos_mask = best_area < INF
    obj_t[0, pos_mask] = 1.0
    ltrb_t[:, pos_mask] = best_ltrb[:, pos_mask]
    return obj_t, ltrb_t


# -------------------------
# CCGO loss (obj alignment to C)
# -------------------------
def ccgo_alignment_loss(
    obj_logits: torch.Tensor,   # [B,1,H,W]
    C_s: torch.Tensor,          # [B,1,H,W] in [0,1]
    bg_mask: Optional[torch.Tensor] = None,  # [B,1,H,W] background=1, else None
) -> torch.Tensor:
    """
    Encourage objectness distribution to align with C_s.
    We typically apply this on background regions to suppress false positives.
    """
    # BCE with soft targets (C_s)
    loss = F.binary_cross_entropy_with_logits(obj_logits, C_s, reduction="none")
    if bg_mask is not None:
        loss = loss * bg_mask
        denom = bg_mask.sum().clamp(min=1.0)
        return loss.sum() / denom
    return loss.mean()


# -------------------------
# Unified Loss Module
# -------------------------
class DetectionLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        preds: List[torch.Tensor],
        C: torch.Tensor,  # [B,1,S,S]
        boxes: Union[List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
          preds: list of [B,5,H,W] length=3
          C: [B,1,S,S]
          boxes:
            - list of length B, each Tensor[Ni,4] (xyxy)
            - OR Tensor[B,Ni,4] with padding (not recommended)

        Returns:
          total_loss, logs dict
        """
        device = preds[0].device
        B = preds[0].shape[0]
        S = self.cfg.img_size

        # normalize boxes container to list[Tensor]
        if isinstance(boxes, torch.Tensor):
            # expect padding with -1 for empty
            boxes_list: List[torch.Tensor] = []
            for b in range(B):
                bb = boxes[b]
                if bb.numel() == 0:
                    boxes_list.append(bb.reshape(0, 4).to(device))
                else:
                    valid = (bb[:, 0] >= 0) & (bb[:, 2] > bb[:, 0]) & (bb[:, 3] > bb[:, 1])
                    boxes_list.append(bb[valid].to(device))
            boxes = boxes_list
        else:
            boxes = [b.to(device) for b in boxes]

        # accumulate
        loss_box_total = torch.tensor(0.0, device=device)
        loss_obj_total = torch.tensor(0.0, device=device)
        loss_cc_total = torch.tensor(0.0, device=device)

        # build per-level targets and compute losses
        for lvl, p in enumerate(preds):
            stride = self.cfg.strides[lvl]
            _, _, H, W = p.shape

            # targets
            obj_t = torch.zeros((B, 1, H, W), device=device)
            ltrb_t = torch.zeros((B, 4, H, W), device=device)

            for b in range(B):
                o, t = assign_targets_center_sampling(
                    boxes_xyxy=boxes[b],
                    H=H, W=W, stride=stride,
                    center_radius=self.cfg.center_radius,
                    device=device,
                )
                obj_t[b:b+1] = o
                ltrb_t[b:b+1] = t

            # predictions
            ltrb_p = p[:, 0:4, :, :]  # [B,4,H,W]
            obj_logit = p[:, 4:5, :, :]  # [B,1,H,W]

            # clamp for stability
            ltrb_p = torch.clamp(ltrb_p, self.cfg.ltrb_min, self.cfg.ltrb_max)

            # --- objectness focal loss (dense)
            loss_obj = focal_bce_with_logits(
                logits=obj_logit,
                targets=obj_t,
                alpha=self.cfg.focal_alpha,
                gamma=self.cfg.focal_gamma,
                reduction="mean",
            )
            loss_obj_total = loss_obj_total + loss_obj

            # --- box loss (only positives)
            pos = (obj_t > 0.5)  # [B,1,H,W]
            num_pos = pos.sum().clamp(min=1.0)

            # decode predicted and target boxes at positive locations (paired)
            if pos.any():
                # grid centers
                cx, cy = _build_grid_centers(H, W, stride, device=device)  # [H,W]
                cx = cx[None, None, :, :].expand(B, 1, H, W)
                cy = cy[None, None, :, :].expand(B, 1, H, W)

                # predicted ltrb are in "cell units" -> convert to pixels
                lp = ltrb_p[:, 0:1] * stride
                tp = ltrb_p[:, 1:2] * stride
                rp = ltrb_p[:, 2:3] * stride
                bp = ltrb_p[:, 3:4] * stride

                # target ltrb in "cell units" -> pixels
                lt = ltrb_t[:, 0:1] * stride
                tt = ltrb_t[:, 1:2] * stride
                rt = ltrb_t[:, 2:3] * stride
                bt = ltrb_t[:, 3:4] * stride

                pred_xyxy = torch.stack(
                    [
                        (cx - lp)[pos],
                        (cy - tp)[pos],
                        (cx + rp)[pos],
                        (cy + bp)[pos],
                    ],
                    dim=1,
                )
                tgt_xyxy = torch.stack(
                    [
                        (cx - lt)[pos],
                        (cy - tt)[pos],
                        (cx + rt)[pos],
                        (cy + bt)[pos],
                    ],
                    dim=1,
                )

                loss_box = ciou_loss(pred_xyxy, tgt_xyxy).mean()
                loss_box_total = loss_box_total + loss_box
            else:
                loss_box = torch.tensor(0.0, device=device)
                loss_box_total = loss_box_total + loss_box

            # --- CCGO loss: align obj logits with downsampled C on background
            C_s = F.interpolate(C, size=(H, W), mode="bilinear", align_corners=False)
            bg_mask = (1.0 - obj_t)  # background=1, positives=0
            loss_cc = ccgo_alignment_loss(obj_logit, C_s, bg_mask=bg_mask)
            loss_cc_total = loss_cc_total + (self.cfg.ccgo_level_weights[lvl] * loss_cc)

        # combine
        total = (
            self.cfg.lambda_box * loss_box_total
            + self.cfg.lambda_obj * loss_obj_total
            + self.cfg.lambda_ccgo * loss_cc_total
        )

        logs = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_box": float(loss_box_total.detach().cpu().item()),
            "loss_obj": float(loss_obj_total.detach().cpu().item()),
            "loss_ccgo": float(loss_cc_total.detach().cpu().item()),
        }
        return total, logs
