# DetectionNetwork/augment.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    img_size: int = 640

    # geometry
    p_affine: float = 0.7
    max_translate: float = 0.10   # relative to S
    max_scale: float = 0.20
    max_rotate: float = 0.0       # 建议先 0，稳定后再尝试 5°

    p_hflip: float = 0.5

    # photometric (img only)
    p_color: float = 0.7
    brightness: float = 0.15
    contrast: float = 0.15

    p_noise: float = 0.3
    noise_sigma: float = 8.0

    # cutout (img only)
    p_cutout: float = 0.3
    cutout_max_frac: float = 0.25

    # box filter
    min_box: float = 2.0


def _ensure_float32(a):
    return a.astype(np.float32, copy=False)


def _clip_xyxy(boxes: np.ndarray, S: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    boxes[:, 0] = np.clip(boxes[:, 0], 0, S - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, S - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, S - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, S - 1)
    return boxes


def _filter_valid(boxes: np.ndarray, min_box: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_box) & (h >= min_box)
    return boxes[keep]


def _warp_boxes_xyxy(boxes: np.ndarray, M: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y1], axis=1),
        np.stack([x2, y2], axis=1),
        np.stack([x1, y2], axis=1),
    ], axis=1).astype(np.float32)

    ones = np.ones((corners.shape[0], 4, 1), dtype=np.float32)
    corners_h = np.concatenate([corners, ones], axis=2)  # [N,4,3]
    warped = corners_h @ M.T  # [N,4,2]

    xs = warped[:, :, 0]
    ys = warped[:, :, 1]
    out = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1).astype(np.float32)
    return out


def _rand_affine(S: int, max_translate: float, max_scale: float, max_rotate: float) -> np.ndarray:
    cx, cy = S * 0.5, S * 0.5
    tx = np.random.uniform(-max_translate, max_translate) * S
    ty = np.random.uniform(-max_translate, max_translate) * S
    sc = np.random.uniform(1.0 - max_scale, 1.0 + max_scale)
    rot = np.random.uniform(-max_rotate, max_rotate)
    M = cv2.getRotationMatrix2D((cx, cy), rot, sc).astype(np.float32)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def _color_jitter(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    img_f = img.astype(np.float32)
    b = np.random.uniform(1.0 - brightness, 1.0 + brightness)
    c = np.random.uniform(1.0 - contrast, 1.0 + contrast)
    img_f = img_f * c + (b - 1.0) * 128.0
    return np.clip(img_f, 0, 255).astype(np.uint8)


def _add_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _cutout(img: np.ndarray, max_frac: float, fill=(114, 114, 114)) -> np.ndarray:
    H, W = img.shape[:2]
    ch = int(np.random.uniform(0.05, max_frac) * H)
    cw = int(np.random.uniform(0.05, max_frac) * W)
    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)
    y1 = max(0, cy - ch // 2)
    y2 = min(H, cy + ch // 2)
    x1 = max(0, cx - cw // 2)
    x2 = min(W, cx + cw // 2)
    out = img.copy()
    out[y1:y2, x1:x2] = np.array(fill, dtype=np.uint8)
    return out


class TrainAugment:
    """
    Signature matches your dataset:
      img_lb: uint8 [S,S,3] RGB
      C_lb:   float32 [1,S,S]
      boxes:  float32 [N,4] xyxy on SxS

    Returns the same three.
    """
    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def __call__(self, img_lb: np.ndarray, C_lb: np.ndarray, boxes_lb: np.ndarray):
        cfg = self.cfg
        S = cfg.img_size

        img_lb = img_lb.astype(np.uint8, copy=False)
        C_lb = _ensure_float32(C_lb)
        boxes_lb = _ensure_float32(boxes_lb)

        # --- affine (img/C/boxes)
        if np.random.rand() < cfg.p_affine:
            M = _rand_affine(S, cfg.max_translate, cfg.max_scale, cfg.max_rotate)

            img_lb = cv2.warpAffine(
                img_lb, M, (S, S),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114),
            )

            C0 = C_lb[0]
            C0 = cv2.warpAffine(
                C0, M, (S, S),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            C_lb = np.clip(C0[None], 0.0, 1.0).astype(np.float32)

            boxes_lb = _warp_boxes_xyxy(boxes_lb, M)
            boxes_lb = _clip_xyxy(boxes_lb, S)
            boxes_lb = _filter_valid(boxes_lb, cfg.min_box)

        # --- hflip (img/C/boxes)
        if np.random.rand() < cfg.p_hflip:
            img_lb = img_lb[:, ::-1, :]
            C_lb = C_lb[:, :, ::-1]
            if boxes_lb.size != 0:
                x1 = boxes_lb[:, 0].copy()
                x2 = boxes_lb[:, 2].copy()
                boxes_lb[:, 0] = (S - 1) - x2
                boxes_lb[:, 2] = (S - 1) - x1

        # --- color (img only)
        if np.random.rand() < cfg.p_color:
            img_lb = _color_jitter(img_lb, cfg.brightness, cfg.contrast)

        # --- noise (img only)
        if np.random.rand() < cfg.p_noise:
            img_lb = _add_noise(img_lb, cfg.noise_sigma)

        # --- cutout (img only)
        if np.random.rand() < cfg.p_cutout:
            img_lb = _cutout(img_lb, cfg.cutout_max_frac)

        return img_lb, C_lb, boxes_lb
