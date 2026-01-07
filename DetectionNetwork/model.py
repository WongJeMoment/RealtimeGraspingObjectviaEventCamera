# src/models/model.py
# -*- coding: utf-8 -*-
"""
CCGODetector:
A self-contained single-stage, anchor-free detector with
objectness + bounding box regression (no class prediction).

Outputs:
  preds = [P3, P4, P5]
  each Px: Tensor[B, 5, H, W]
    channels = (l, t, r, b, obj_logit)

Decode:
  model.decode(preds, conf_thres, topk, img_size) -> boxes/scores
  (No NMS inside decode)

This file intentionally avoids any YOLO-specific naming.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic building blocks
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = ConvBNAct(ch, ch, 3, 1)
        self.c2 = ConvBNAct(ch, ch, 3, 1)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class CSPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n: int = 1):
        super().__init__()
        hidden = out_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden, 1, 1, 0)
        self.conv2 = ConvBNAct(in_ch, hidden, 1, 1, 0)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n)])
        self.merge = ConvBNAct(hidden * 2, out_ch, 1, 1, 0)

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.merge(torch.cat([y1, y2], dim=1))


# -------------------------
# Backbone
# -------------------------
class FeatureBackbone(nn.Module):
    """
    Produces multi-scale features at strides ~8,16,32.
    """
    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base, 3, 2),
            ConvBNAct(base, base, 3, 1),
        )

        self.stage2 = nn.Sequential(
            ConvBNAct(base, base * 2, 3, 2),
            CSPBlock(base * 2, base * 2, n=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(base * 2, base * 4, 3, 2),
            CSPBlock(base * 4, base * 4, n=2),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(base * 4, base * 8, 3, 2),
            CSPBlock(base * 8, base * 8, n=2),
        )
        self.stage5 = nn.Sequential(
            ConvBNAct(base * 8, base * 16, 3, 2),
            CSPBlock(base * 16, base * 16, n=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        f3 = self.stage3(x)  # stride ~8
        f4 = self.stage4(f3) # stride ~16
        f5 = self.stage5(f4) # stride ~32
        return f3, f4, f5


# -------------------------
# Feature Pyramid
# -------------------------
class FeaturePyramid(nn.Module):
    def __init__(self, c3: int, c4: int, c5: int, out_ch: int = 128):
        super().__init__()
        self.l3 = ConvBNAct(c3, out_ch, 1, 1, 0)
        self.l4 = ConvBNAct(c4, out_ch, 1, 1, 0)
        self.l5 = ConvBNAct(c5, out_ch, 1, 1, 0)

        self.smooth3 = ConvBNAct(out_ch, out_ch, 3, 1)
        self.smooth4 = ConvBNAct(out_ch, out_ch, 3, 1)
        self.smooth5 = ConvBNAct(out_ch, out_ch, 3, 1)

    def forward(self, f3, f4, f5):
        p3 = self.l3(f3)
        p4 = self.l4(f4)
        p5 = self.l5(f5)

        p4 = self.smooth4(p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3 = self.smooth3(p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))
        p5 = self.smooth5(p5)

        return p3, p4, p5


# -------------------------
# Detection Head
# -------------------------
class ObjectnessHead(nn.Module):
    """
    Output per level: [B, 5, H, W] = (l,t,r,b,obj_logit)
    NOTE: l,t,r,b are distances in "cell units" (to be multiplied by stride at decode time).
    """
    def __init__(self, in_ch: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, hidden, 3, 1),
            ConvBNAct(hidden, hidden, 3, 1),
        )
        self.pred = nn.Conv2d(hidden, 5, 1)

        nn.init.normal_(self.pred.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.pred.bias[:] = 0.0
            self.pred.bias[4] = -4.0  # low objectness prior

    def forward(self, x):
        return self.pred(self.conv(x))


# -------------------------
# Full Detector
# -------------------------
@dataclass
class DetectorConfig:
    in_channels: int = 3
    base: int = 32
    fpn_channels: int = 128
    head_channels: int = 128
    strides: Tuple[int, int, int] = (8, 16, 32)


class CCGODetector(nn.Module):
    """
    Color-Consistency Guided Objectness Detector.
    """
    def __init__(self, cfg: DetectorConfig = DetectorConfig()):
        super().__init__()
        self.cfg = cfg

        self.backbone = FeatureBackbone(cfg.in_channels, cfg.base)

        c3 = cfg.base * 4
        c4 = cfg.base * 8
        c5 = cfg.base * 16
        self.fpn = FeaturePyramid(c3, c4, c5, cfg.fpn_channels)

        self.head3 = ObjectnessHead(cfg.fpn_channels, cfg.head_channels)
        self.head4 = ObjectnessHead(cfg.fpn_channels, cfg.head_channels)
        self.head5 = ObjectnessHead(cfg.fpn_channels, cfg.head_channels)

    @property
    def strides(self) -> Tuple[int, int, int]:
        return self.cfg.strides

    def forward(self, x) -> List[torch.Tensor]:
        f3, f4, f5 = self.backbone(x)
        p3, p4, p5 = self.fpn(f3, f4, f5)
        return [
            self.head3(p3),
            self.head4(p4),
            self.head5(p5),
        ]

    @torch.no_grad()
    def decode(
        self,
        preds: List[torch.Tensor],
        conf_thres: float = 0.25,
        topk: int = 300,
        img_size: Optional[int] = None,
        clip: bool = True,
        relu_ltrb: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode raw dense outputs into image-space boxes (xyxy) + scores.

        Args:
            preds: list of [B,5,H,W], channels=(l,t,r,b,obj_logit)
            conf_thres: threshold on sigmoid(obj_logit)
            topk: keep top-k scored boxes per image after thresholding (across all levels)
            img_size: if not None, clip boxes into [0, img_size-1]
            clip: whether to clip boxes
            relu_ltrb: apply relu to ltrb to enforce non-negative distances

        Returns:
            results: list length B, each dict:
              {
                "boxes": Tensor[N,4] in xyxy (float, pixel coords),
                "scores": Tensor[N]
              }
            NOTE: This does NOT perform NMS.
        """
        device = preds[0].device
        B = preds[0].shape[0]
        strides = self.cfg.strides

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            boxes_all = []
            scores_all = []

            for lvl, p in enumerate(preds):
                stride = strides[lvl]
                p_b = p[b]  # [5,H,W]
                ltrb = p_b[0:4]  # [4,H,W]
                if relu_ltrb:
                    ltrb = torch.relu(ltrb)

                obj_logit = p_b[4]              # [H,W]
                score = torch.sigmoid(obj_logit)  # [H,W]
                H, W = score.shape

                ys, xs = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing="ij",
                )
                cx = (xs + 0.5) * stride
                cy = (ys + 0.5) * stride

                # l,t,r,b: cell units -> pixels
                l = ltrb[0] * stride
                t = ltrb[1] * stride
                r = ltrb[2] * stride
                bt = ltrb[3] * stride

                x1 = cx - l
                y1 = cy - t
                x2 = cx + r
                y2 = cy + bt

                keep = score > conf_thres
                if keep.sum() == 0:
                    continue

                boxes = torch.stack([x1[keep], y1[keep], x2[keep], y2[keep]], dim=1)
                sc = score[keep]
                boxes_all.append(boxes)
                scores_all.append(sc)

            if len(boxes_all) == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                })
                continue

            boxes = torch.cat(boxes_all, dim=0)
            scores = torch.cat(scores_all, dim=0)

            if clip and (img_size is not None):
                boxes[:, 0] = boxes[:, 0].clamp(0, img_size - 1)
                boxes[:, 1] = boxes[:, 1].clamp(0, img_size - 1)
                boxes[:, 2] = boxes[:, 2].clamp(0, img_size - 1)
                boxes[:, 3] = boxes[:, 3].clamp(0, img_size - 1)

            if topk is not None and boxes.shape[0] > topk:
                idx = scores.topk(topk, largest=True).indices
                boxes = boxes[idx]
                scores = scores[idx]

            results.append({"boxes": boxes, "scores": scores})

        return results


# -------------------------
# Sanity check + decode demo
# -------------------------
if __name__ == "__main__":
    model = CCGODetector(DetectorConfig(in_channels=3))
    x = torch.randn(1, 3, 640, 640)
    preds = model(x)
    for o in preds:
        print(o.shape)

    res = model.decode(preds, conf_thres=0.25, topk=50, img_size=640)
    print("Decoded boxes:", res[0]["boxes"].shape, "scores:", res[0]["scores"].shape)
