# src/data/augment.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    img_size: int = 640

    # affine
    p_affine: float = 0.7
    max_translate: float = 0.10   # relative to img_size
    max_scale: float = 0.20       # scale jitter
    max_rotate: float = 0.0       # 建议先 0 或很小，比如 5°

    # flip
    p_hflip: float = 0.5

    # photometric
    p_color: float = 0.7
    brightness: float = 0.15
    contrast: float = 0.15

    # noise
    p_noise: float = 0.3
    noise_sigma: float = 8.0

    # cutout
    p_cutout: float = 0.3
    cutout_max_frac: float = 0.25


def _clip_boxes_xyxy(boxes: np.ndarray, S: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, S - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, S - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, S - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, S - 1)
    return boxes


def _filter_valid_boxes(boxes: np.ndarray, min_size: float = 2.0) -> np.ndarray:
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    return boxes[keep]


def _warp_affine(img: np.ndarray, M: np.ndarray, S: int, border_val: Tuple[int, int, int]):
    return cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)


def _warp_affine_gray(im: np.ndarray, M: np.ndarray, S: int, border_val: float):
    return cv2.warpAffine(im, M, (S, S), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_val))


def _warp_boxes_xyxy(boxes: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply affine transform to xyxy boxes by warping 4 corners then taking min/max.
    boxes: [N,4]
    """
    if boxes.size == 0:
        return boxes

    # corners: [N,4,2]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y1], axis=1),
        np.stack([x2, y2], axis=1),
        np.stack([x1, y2], axis=1),
    ], axis=1).astype(np.float32)

    # to homogeneous: [N,4,3]
    ones = np.ones((corners.shape[0], corners.shape[1], 1), dtype=np.float32)
    corners_h = np.concatenate([corners, ones], axis=2)

    # M: [2,3], apply: [N,4,2]
    warped = corners_h @ M.T

    x = warped[:, :, 0]
    y = warped[:, :, 1]
    new_boxes = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], axis=1)
    return new_boxes.astype(np.float32)


def _rand_affine(S: int, max_translate: float, max_scale: float, max_rotate: float):
    # center
    cx, cy = S * 0.5, S * 0.5

    # random params
    tx = np.random.uniform(-max_translate, max_translate) * S
    ty = np.random.uniform(-max_translate, max_translate) * S
    sc = np.random.uniform(1.0 - max_scale, 1.0 + max_scale)
    rot = np.random.uniform(-max_rotate, max_rotate)

    # rotation+scale around center
    M = cv2.getRotationMatrix2D((cx, cy), rot, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    return M.astype(np.float32)


def _color_jitter(img: np.ndarray, brightness: float, contrast: float):
    # img: uint8 RGB
    img_f = img.astype(np.float32)

    b = np.random.uniform(1.0 - brightness, 1.0 + brightness)
    c = np.random.uniform(1.0 - contrast, 1.0 + contrast)

    img_f = img_f * c
    img_f = img_f + (b - 1.0) * 128.0
    img_f = np.clip(img_f, 0, 255).astype(np.uint8)
    return img_f


def _add_noise(img: np.ndarray, sigma: float):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _cutout(img: np.ndarray, max_frac: float, fill: Tuple[int, int, int] = (114, 114, 114)):
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
    def __init__(self, img_size=640, p_affine=0.7, p_hflip=0.5,
                 max_translate=0.1, max_scale=0.2):
        self.S = img_size
        self.p_affine = p_affine
        self.p_hflip = p_hflip
        self.max_translate = max_translate
        self.max_scale = max_scale

    def __call__(self, img_lb, C_lb, boxes_lb):
        # img_lb: uint8 [S,S,3] RGB
        # C_lb: float32 [1,S,S]
        # boxes_lb: float32 [N,4]
        S = self.S

        # affine (no rotation)
        if np.random.rand() < self.p_affine:
            tx = np.random.uniform(-self.max_translate, self.max_translate) * S
            ty = np.random.uniform(-self.max_translate, self.max_translate) * S
            sc = np.random.uniform(1.0 - self.max_scale, 1.0 + self.max_scale)

            M = np.array([[sc, 0, tx],
                          [0, sc, ty]], dtype=np.float32)

            img_lb = cv2.warpAffine(img_lb, M, (S, S), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(114,114,114))

            C0 = C_lb[0]
            C0 = cv2.warpAffine(C0, M, (S, S), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            C_lb = C0[None].astype(np.float32)

            if boxes_lb.size != 0:
                # warp 4 corners
                x1, y1, x2, y2 = boxes_lb[:,0], boxes_lb[:,1], boxes_lb[:,2], boxes_lb[:,3]
                corners = np.stack([
                    np.stack([x1,y1],1),
                    np.stack([x2,y1],1),
                    np.stack([x2,y2],1),
                    np.stack([x1,y2],1),
                ], axis=1).astype(np.float32)
                ones = np.ones((corners.shape[0], 4, 1), dtype=np.float32)
                ch = np.concatenate([corners, ones], axis=2)
                w = ch @ M.T
                xs, ys = w[:,:,0], w[:,:,1]
                boxes_lb = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1).astype(np.float32)

        # hflip
        if np.random.rand() < self.p_hflip:
            img_lb = img_lb[:, ::-1, :]
            C_lb = C_lb[:, :, ::-1]
            if boxes_lb.size != 0:
                x1 = boxes_lb[:, 0].copy()
                x2 = boxes_lb[:, 2].copy()
                boxes_lb[:, 0] = (S - 1) - x2
                boxes_lb[:, 2] = (S - 1) - x1

        return img_lb, C_lb, boxes_lb
