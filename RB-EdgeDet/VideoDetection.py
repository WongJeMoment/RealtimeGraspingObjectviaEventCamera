# VisualVideo.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import torch

# =========================
# ✅ 配置区（只看这里）
# =========================
VIDEO_PATH = "/home/wangzhe/2026/IROS/Dataset/VideoNew/apple3.mp4"     # ✅ 改成你的视频路径
OUT_PATH = "./vis_results/out_detect.mp4"  # ✅ 输出视频
CKPT_PATH = "/home/wangzhe/2026/IROS/RB-EdgeDet/checkpoints/ccgo_detector_best.pth"

IMG_SIZE = 640

CONF_THRES = 0.25
IOU_THRES = 0.6
TOPK = 300
MIN_AREA = 32 * 32

SHOW = False          # True：边跑边弹窗显示（远程服务器一般关掉）
MAX_FRAMES = -1       # -1 表示不限制；比如 500 只跑前 500 帧

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Imports from your project
# =========================
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


def filter_small_boxes_xyxy(boxes: torch.Tensor, scores: torch.Tensor, min_area: float):
    if boxes.numel() == 0:
        return boxes, scores
    wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]
    keep = area >= min_area
    return boxes[keep], scores[keep]


def preprocess_bgr_to_tensor(frame_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    """
    frame_bgr: HxWx3 uint8
    return: 1x3xSxS float32 in [0,1]
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = frame_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1,3,S,S
    return x


def draw_boxes_bgr(frame_bgr: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray):
    out = frame_bgr.copy()
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
        cv2.putText(
            out, f"{scores[i]:.2f}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    return out


@torch.no_grad()
def main():
    print("🚀 Video Detection Visualizer")
    print("Device:", DEVICE)
    print("Video:", VIDEO_PATH)

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

    # -------- Video IO --------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps if fps > 0 else 25.0, (w0, h0))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if MAX_FRAMES > 0 and frame_idx > MAX_FRAMES:
            break

        # -------- preprocess & inference --------
        x = preprocess_bgr_to_tensor(frame, IMG_SIZE).to(DEVICE)

        preds = model(x)
        decoded = model.decode(
            preds,
            conf_thres=CONF_THRES,
            topk=TOPK,
            img_size=IMG_SIZE,
        )[0]

        pb = decoded["boxes"]
        ps = decoded["scores"]

        # ✅ 先过滤小框
        pb, ps = filter_small_boxes_xyxy(pb, ps, MIN_AREA)

        # ✅ 再 NMS
        keep = nms_xyxy(pb, ps, IOU_THRES)
        pb = pb[keep]
        ps = ps[keep]

        # ✅ 再 top-1
        if ps.numel() > 0:
            top1 = torch.argmax(ps)
            pb = pb[top1:top1 + 1]
            ps = ps[top1:top1 + 1]

        # -------- map boxes back to original frame size --------
        # decode 的坐标是基于 IMG_SIZE 的 (640x640)，需要缩放回原视频大小
        pb_np = pb.detach().cpu().numpy()
        ps_np = ps.detach().cpu().numpy()

        if pb_np.shape[0] > 0:
            sx = w0 / float(IMG_SIZE)
            sy = h0 / float(IMG_SIZE)
            pb_np[:, [0, 2]] *= sx
            pb_np[:, [1, 3]] *= sy

        # -------- draw & write --------
        vis = frame
        if pb_np.shape[0] > 0:
            vis = draw_boxes_bgr(vis, pb_np, ps_np)

        cv2.putText(
            vis,
            f"frame:{frame_idx}  pred:{pb_np.shape[0]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )

        writer.write(vis)

        if SHOW:
            cv2.imshow("det", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    if SHOW:
        cv2.destroyAllWindows()

    print("✅ Done. Saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
