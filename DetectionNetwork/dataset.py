# src/data/dataset.py
# -*- coding: utf-8 -*-
"""
Detection dataset with Color-Consistency Map C(x,y) for CCGO.

What this dataset returns (per sample):
  - img:  FloatTensor [3, H, W] in [0,1]  (RGB)
  - C:    FloatTensor [1, H, W] in [0,1]  (consistency map)
  - boxes: FloatTensor [N, 4] (xyxy, pixel coords AFTER resize/pad/augment)
  - meta: dict (path, original size, resize info, etc.)

Annotation format:
  Default supports YOLO txt format (class x_center y_center w h), normalized in [0,1].
  Class id is ignored (you do "no class"), so any integer is fine.

Folder layout expected (YOLO-like):
  images/
    0001.jpg
    0002.png
  labels/
    0001.txt
    0002.txt

You can pass:
  - images_dir: path to images
  - labels_dir: path to labels
  - img_size:   int, e.g., 640
  - augment:    optional callable that applies geometric transforms consistently to (img, C, boxes)
               signature: img, C, boxes = augment(img, C, boxes)

Dependencies:
  pip install torch opencv-python numpy
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from DetectionNetwork.consistency import ConsistencyConfig, build_consistency_map_1chw


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetConfig:
    images_dir: str
    labels_dir: str
    img_size: int = 640  # output square size
    input_is_bgr: bool = True  # cv2.imread -> BGR

    # letterbox padding color (RGB) - we keep img as RGB after conversion
    pad_color: Tuple[int, int, int] = (114, 114, 114)

    # consistency map config
    consistency: ConsistencyConfig = ConsistencyConfig(input_color="BGR")

    # if label missing, allow empty targets
    allow_empty: bool = True

    # clamp boxes to image bounds after transforms
    clip_boxes: bool = True


def _list_images(images_dir: Union[str, Path]) -> List[Path]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    paths = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    paths.sort()
    if len(paths) == 0:
        raise RuntimeError(f"No images found under: {images_dir}")
    return paths


def _yolo_label_path(img_path: Path, labels_dir: Union[str, Path]) -> Path:
    labels_dir = Path(labels_dir)
    return labels_dir / (img_path.stem + ".txt")


def _read_yolo_labels(label_path: Path) -> np.ndarray:
    """
    Returns boxes in YOLO normalized format (xc, yc, w, h), shape [N,4] float32.
    Ignores class id if present.
    """
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    lines = label_path.read_text().strip().splitlines()
    if len(lines) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = []
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) < 5:
            # allow malformed lines quietly
            continue
        # parts: cls xc yc w h
        xc, yc, w, h = map(float, parts[1:5])
        boxes.append([xc, yc, w, h])

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def _yolo_to_xyxy(boxes_yolo: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    boxes_yolo: [N,4] normalized (xc,yc,w,h) in [0,1]
    returns:    [N,4] absolute xyxy in pixels
    """
    if boxes_yolo.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    xc = boxes_yolo[:, 0] * w
    yc = boxes_yolo[:, 1] * h
    bw = boxes_yolo[:, 2] * w
    bh = boxes_yolo[:, 3] * h

    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _clip_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def letterbox(
    img_rgb: np.ndarray,
    C_1hw: np.ndarray,
    boxes_xyxy: np.ndarray,
    new_size: int,
    pad_color_rgb: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Resize + pad to square (new_size x new_size) while keeping aspect ratio.
    Applies same transform to img, C, boxes.

    img_rgb: HxWx3 uint8
    C_1hw:   1xHxW float32 in [0,1]
    boxes_xyxy: Nx4 float32 (pixel coords on original image)

    Returns:
      img_out: new_size x new_size x 3 uint8
      C_out:   1 x new_size x new_size float32
      boxes_out: Nx4 float32 (pixel coords on output)
      info: dict with scale & padding
    """
    H, W = img_rgb.shape[:2]
    assert C_1hw.shape[1] == H and C_1hw.shape[2] == W, "C must match image size"

    scale = min(new_size / H, new_size / W)
    nh, nw = int(round(H * scale)), int(round(W * scale))

    # resize
    img_rs = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    C_rs = cv2.resize(C_1hw[0], (nw, nh), interpolation=cv2.INTER_LINEAR)[None, :, :]

    # pad
    pad_h = new_size - nh
    pad_w = new_size - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_out = cv2.copyMakeBorder(
        img_rs, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=pad_color_rgb
    )
    C_out = np.pad(
        C_rs,
        pad_width=((0, 0), (top, bottom), (left, right)),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32)

    # boxes transform
    boxes = boxes_xyxy.copy()
    if boxes.size != 0:
        boxes *= scale
        boxes[:, [0, 2]] += left
        boxes[:, [1, 3]] += top

    info = {
        "orig_hw": (H, W),
        "resized_hw": (nh, nw),
        "scale": scale,
        "pad": (top, left, bottom, right),  # (top,left,bottom,right)
        "out_hw": (new_size, new_size),
    }
    return img_out, C_out, boxes, info


class DetectionCCGODataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        augment: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ):
        self.cfg = cfg
        self.augment = augment
        self.image_paths = _list_images(cfg.images_dir)

        self.labels_dir = Path(cfg.labels_dir)
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"labels_dir not found: {self.labels_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_image_rgb(self, path: Path) -> np.ndarray:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read image: {path}")

        # cv2.imread gives BGR. Convert to RGB for model input.
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im_rgb

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        label_path = _yolo_label_path(img_path, self.labels_dir)

        # --- read image
        img_rgb = self._read_image_rgb(img_path)
        H, W = img_rgb.shape[:2]

        # --- read labels (YOLO normalized -> xyxy pixel)
        boxes_yolo = _read_yolo_labels(label_path)
        boxes_xyxy = _yolo_to_xyxy(boxes_yolo, W, H)

        if (boxes_xyxy.size == 0) and (not self.cfg.allow_empty):
            raise RuntimeError(f"Empty labels not allowed but found: {label_path}")

        # --- build consistency map C (based on original image; use BGR config internally)
        # Our build_consistency_map expects cfg.input_color; easiest is to feed BGR from cv2
        # But we already converted to RGB for model. Convert back to BGR for C builder if needed.
        if self.cfg.consistency.input_color.upper() == "BGR":
            img_for_C = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            img_for_C = img_rgb

        C_1hw = build_consistency_map_1chw(img_for_C, self.cfg.consistency)  # (1,H,W), float32 0..1

        # --- letterbox to fixed size
        img_lb, C_lb, boxes_lb, info = letterbox(
            img_rgb=img_rgb,
            C_1hw=C_1hw,
            boxes_xyxy=boxes_xyxy,
            new_size=self.cfg.img_size,
            pad_color_rgb=self.cfg.pad_color,
        )

        # --- optional augment (must keep geometry consistent)
        if self.augment is not None:
            img_lb, C_lb, boxes_lb = self.augment(img_lb, C_lb, boxes_lb)

        # --- clip boxes
        if self.cfg.clip_boxes:
            boxes_lb = _clip_xyxy(boxes_lb, self.cfg.img_size, self.cfg.img_size)

        # remove invalid boxes (x2<=x1 or y2<=y1)
        if boxes_lb.size != 0:
            w_box = boxes_lb[:, 2] - boxes_lb[:, 0]
            h_box = boxes_lb[:, 3] - boxes_lb[:, 1]
            keep = (w_box > 1.0) & (h_box > 1.0)
            boxes_lb = boxes_lb[keep]

        # --- to torch
        img_t = torch.from_numpy(img_lb.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,H,W]
        C_t = torch.from_numpy(C_lb.astype(np.float32))  # [1,H,W]
        boxes_t = torch.from_numpy(boxes_lb.astype(np.float32))  # [N,4]

        meta = {
            "img_path": str(img_path),
            "label_path": str(label_path),
            "orig_hw": info["orig_hw"],
            "scale": info["scale"],
            "pad": info["pad"],
            "out_hw": info["out_hw"],
        }

        return {
            "img": img_t,
            "C": C_t,
            "boxes": boxes_t,
            "meta": meta,
        }
