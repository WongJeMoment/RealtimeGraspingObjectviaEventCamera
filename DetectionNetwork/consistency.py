# src/color/consistency.py
# -*- coding: utf-8 -*-
"""
Color Consistency Map (CCGO) builder.

Goal:
  Given an image, build a soft "color-consistency" map C(x,y) in [0,1]
  that highlights structured red/blue clusters and suppresses sparse noise.

Design:
  1) Convert to HSV
  2) Compute soft red/blue probability (no hard threshold)
  3) Turn "probability" into "consistency" via local density (blur)
  4) Optional contrast stretching / gamma

Dependencies:
  pip install opencv-python numpy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import cv2
import numpy as np


@dataclass
class ConsistencyConfig:
    # input format
    input_color: Literal["RGB", "BGR"] = "RGB"

    # HSV soft scoring parameters
    # Hue is in [0, 180] in OpenCV. Red wraps around 0/180.
    # We'll use circular distance for red and normal distance for blue.
    red_centers: Tuple[int, int] = (0, 180)   # treat as equivalent centers
    blue_center: int = 120

    # how wide the hue peak is (bigger = broader acceptance)
    sigma_h_red: float = 10.0
    sigma_h_blue: float = 14.0

    # saturation/value gating: suppress gray/dark areas
    sat_gate_center: float = 60.0   # in [0,255]
    val_gate_center: float = 60.0
    sat_gate_width: float = 25.0
    val_gate_width: float = 25.0

    # local density smoothing to convert sparse pixels -> cluster map
    blur_ksize: int = 21            # must be odd; larger => smoother/denser
    blur_sigma: float = 0.0         # 0 uses OpenCV default from ksize

    # optional "boost"
    # percentile stretch makes high-response regions more separable
    stretch_percentiles: Tuple[float, float] = (5.0, 99.0)  # (low, high)
    gamma: float = 1.0              # >1 suppress mid values, <1 boost mid values

    # optional cleanup
    # remove tiny speckles by small blur + opening on a soft map (kept soft)
    enable_soft_cleanup: bool = True
    cleanup_ksize: int = 3          # small morph kernel (odd)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _gate(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Smooth gate in [0,1]. Higher when x >= center-ish.
    width controls softness.
    """
    # x, center in same scale; width ~ transition scale
    return _sigmoid((x - center) / max(1e-6, width))


def _circular_hue_dist(h: np.ndarray, center: float) -> np.ndarray:
    """
    Hue distance on a circle for OpenCV hue range [0,180].
    Returns distance in [0, 90].
    """
    d = np.abs(h - center)
    return np.minimum(d, 180.0 - d)


def _gaussian_score(dist: np.ndarray, sigma: float) -> np.ndarray:
    """
    Convert distance -> score in (0,1], Gaussian-like.
    """
    s2 = max(1e-6, sigma) ** 2
    return np.exp(-(dist ** 2) / (2.0 * s2))


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def build_consistency_map(img: np.ndarray, cfg: Optional[ConsistencyConfig] = None) -> np.ndarray:
    """
    Build C(x,y) in [0,1], float32, shape (H,W).

    Args:
      img: HxWx3 uint8 image
      cfg: ConsistencyConfig

    Returns:
      C: HxW float32 in [0,1]
    """
    if cfg is None:
        cfg = ConsistencyConfig()

    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={None if img is None else img.shape}")

    if img.dtype != np.uint8:
        # tolerate float images in [0,1] or [0,255]
        img_f = img.astype(np.float32)
        if img_f.max() <= 1.5:
            img_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img_f, 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    if cfg.input_color.upper() == "RGB":
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    elif cfg.input_color.upper() == "BGR":
        bgr = img_u8
    else:
        raise ValueError("cfg.input_color must be 'RGB' or 'BGR'")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)  # 0..180
    s = hsv[:, :, 1].astype(np.float32)  # 0..255
    v = hsv[:, :, 2].astype(np.float32)  # 0..255

    # Soft hue scores
    # Red: wrap-around: take best of two centers (0 and 180 are the same hue)
    d_red0 = _circular_hue_dist(h, float(cfg.red_centers[0]))
    d_red1 = _circular_hue_dist(h, float(cfg.red_centers[1]))
    red_score = np.maximum(_gaussian_score(d_red0, cfg.sigma_h_red),
                           _gaussian_score(d_red1, cfg.sigma_h_red))

    # Blue: normal circular distance also works fine
    d_blue = _circular_hue_dist(h, float(cfg.blue_center))
    blue_score = _gaussian_score(d_blue, cfg.sigma_h_blue)

    # Saturation/Value gating (suppress gray/dark background)
    g_s = _gate(s, cfg.sat_gate_center, cfg.sat_gate_width)
    g_v = _gate(v, cfg.val_gate_center, cfg.val_gate_width)
    gate = g_s * g_v

    # Soft red/blue probability-like map
    p_rb = np.clip((red_score + blue_score) * 0.5, 0.0, 1.0) * gate

    # Convert probability -> consistency via local density smoothing
    k = _ensure_odd(cfg.blur_ksize)
    if k > 1:
        c = cv2.GaussianBlur(p_rb.astype(np.float32), (k, k), cfg.blur_sigma)
    else:
        c = p_rb.astype(np.float32)

    # Optional soft cleanup: slight opening on a thresholded proxy, then blend back
    # Keeps output soft but reduces tiny isolated speckles.
    if cfg.enable_soft_cleanup:
        kk = _ensure_odd(cfg.cleanup_ksize)
        if kk > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
            # create a soft proxy mask from c, but not too harsh
            proxy = (c > np.percentile(c, 60)).astype(np.uint8) * 255
            proxy = cv2.morphologyEx(proxy, cv2.MORPH_OPEN, kernel, iterations=1)
            proxy_f = (proxy.astype(np.float32) / 255.0)
            # blend: keep c where proxy says "likely cluster"
            c = c * (0.35 + 0.65 * proxy_f)

    # Percentile stretch to [0,1]
    lo_p, hi_p = cfg.stretch_percentiles
    lo = float(np.percentile(c, lo_p))
    hi = float(np.percentile(c, hi_p))
    if hi <= lo + 1e-6:
        c01 = np.clip(c, 0.0, 1.0)
    else:
        c01 = (c - lo) / (hi - lo)
        c01 = np.clip(c01, 0.0, 1.0)

    # Gamma adjustment
    if abs(cfg.gamma - 1.0) > 1e-6:
        # keep numerical stability
        c01 = np.clip(c01, 0.0, 1.0) ** float(cfg.gamma)

    return c01.astype(np.float32)


def build_consistency_map_1chw(img: np.ndarray, cfg: Optional[ConsistencyConfig] = None) -> np.ndarray:
    """
    Convenience wrapper returning (1,H,W) float32 tensor-like ndarray.
    """
    c = build_consistency_map(img, cfg)
    return c[None, :, :].astype(np.float32)


# -----------------------------
# Quick manual test
# -----------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate color consistency map C(x,y)")
    parser.add_argument(
        "--img",
        type=str,
        default="/home/wangzhe/2026/IROS/MyDataset/IMG/apple/1.png",
        help="input image path (default: a demo image)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./consistency_vis.png",
        help="output visualization path",
    )
    parser.add_argument(
        "--bgr",
        action="store_true",
        help="treat input image as BGR (recommended for cv2.imread)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.img):
        raise FileNotFoundError(f"Image not found: {args.img}")

    # OpenCV always reads BGR
    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {args.img}")

    cfg = ConsistencyConfig(
        input_color="BGR" if args.bgr or True else "RGB"
    )

    C = build_consistency_map(img, cfg)

    vis = (C * 255.0).astype(np.uint8)
    cv2.imwrite(args.out, vis)

    print("=" * 50)
    print("Color Consistency Map generated successfully")
    print(f"Input : {args.img}")
    print(f"Output: {args.out}")
    print(f"Range : min={C.min():.4f}, max={C.max():.4f}")
    print("=" * 50)

