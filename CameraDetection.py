# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple

import sys
import time
import cv2
import numpy as np
import torch

import config as cfg
from DetectionNetwork.model import CCGODetector, DetectorConfig
from TrackingNetwork.model import PMD_TSD_Box

# ============ Metavision (按 sync.py) ============
sys.path.append("/usr/local/local/lib/python3.8/dist-packages/")
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent

# =========================
# 可配置区
# =========================
# 如果为空 "" -> 直接用相机实时流；如果是 RAW/DAT/HDF5 文件路径就读文件
EVENT_INPUT = ""   # e.g. "" or "/path/to/file.raw" or camera serial

# 输出录像（按 V 开始/停止）
OUT_VIDEO_PATH = "out_live_track.avi"
SAVE_FPS = 25.0

# Detector
DET_CKPT_PATH = "/home/wangzhe/2026/IROS/RB-EdgeDet/checkpoints/ccgo_detector_best.pth"
DET_IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.6
TOPK = 300
MIN_AREA = 32 * 32

# Tracker
TRACK_H, TRACK_W = cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH
TRACK_CKPT_PATH = str(cfg.MODEL_SAVE_PATH)
DEVICE = cfg.DEVICE

# -------------------------
# 重检策略
# -------------------------
REDETECT_EVERY = 1          # 每 N 帧强制重检一次；0 表示关闭
REDETECT_CONF_THRES = 0.55  # tracker conf 低于该值触发重检
PRINT_DEBUG_EVERY = 5       # 每 N 帧打印一次（0 关闭）


# =========================
# Utils
# =========================
@torch.no_grad()
def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thres=0.6) -> torch.Tensor:
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


def draw_box(img_bgr: np.ndarray, box_xyxy: np.ndarray, color=(0, 0, 255), text: str = "") -> np.ndarray:
    out = img_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(out, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def xyxy_rescale(box_xyxy: np.ndarray, from_wh: Tuple[int, int], to_wh: Tuple[int, int]) -> np.ndarray:
    fw, fh = from_wh
    tw, th = to_wh
    sx, sy = tw / fw, th / fh
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


def frame_to_tensor(frame_bgr: np.ndarray, out_hw: Tuple[int, int], bgr2rgb: bool = True) -> torch.Tensor:
    h, w = out_hw
    img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    if bgr2rgb:
        img = img[:, :, ::-1].copy()  # BGR -> RGB
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def clamp_fix_xyxy(box_xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = float(np.clip(x1, 0, w - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def maybe_norm_to_xyxy(pred4: np.ndarray, w: int, h: int):
    pmin, pmax = float(pred4.min()), float(pred4.max())
    is_norm = (pmin >= -0.2) and (pmax <= 1.5)
    if is_norm:
        x1, y1, x2, y2 = pred4.astype(np.float32)
        return np.array([x1 * w, y1 * h, x2 * w, y2 * h], dtype=np.float32), True
    return pred4.astype(np.float32), False


# =========================
# Detector Wrapper
# =========================
class Detector:
    def __init__(self, ckpt_path: str):
        self.model = CCGODetector(
            DetectorConfig(in_channels=3, base=32, fpn_channels=128, head_channels=128)
        ).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        self.model.eval()

    @torch.no_grad()
    def infer_top1_box_xyxy_detspace(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        inp = frame_to_tensor(frame_bgr, (DET_IMG_SIZE, DET_IMG_SIZE), bgr2rgb=True).to(DEVICE)
        preds = self.model(inp)
        decoded = self.model.decode(preds, conf_thres=CONF_THRES, topk=TOPK, img_size=DET_IMG_SIZE)

        pb = decoded[0]["boxes"]
        ps = decoded[0]["scores"]
        pb, ps = filter_small_boxes_xyxy(pb, ps, min_area=MIN_AREA)
        keep = nms_xyxy(pb, ps, IOU_THRES)
        pb, ps = pb[keep], ps[keep]

        if ps.numel() == 0:
            return None
        top1 = torch.argmax(ps)
        return pb[top1].detach().cpu().numpy().astype(np.float32)


# =========================
# Tracker Wrapper
# =========================
class VideoTracker:
    def __init__(self, ckpt_path: str):
        self.model = PMD_TSD_Box().to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.reset()

    def reset(self):
        self.prev_img_t: Optional[torch.Tensor] = None
        self.prev_box: Optional[torch.Tensor] = None

    @torch.no_grad()
    def init(self, frame_bgr: np.ndarray, init_box_xyxy_trackspace: np.ndarray):
        img_t = frame_to_tensor(frame_bgr, (TRACK_H, TRACK_W), bgr2rgb=True).to(DEVICE)
        box = init_box_xyxy_trackspace.astype(np.float32).reshape(1, 4)
        self.prev_img_t = img_t
        self.prev_box = torch.from_numpy(box).to(DEVICE)

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray):
        if self.prev_img_t is None or self.prev_box is None:
            return None, 0.0, False

        img_t = frame_to_tensor(frame_bgr, (TRACK_H, TRACK_W), bgr2rgb=True).to(DEVICE)
        img_tm1 = self.prev_img_t
        box_tm1 = self.prev_box

        pred_box_t, conf = self.model(img_t, img_tm1, box_tm1)

        pred = pred_box_t.detach().view(-1).cpu().numpy().astype(np.float32)
        confv = float(conf.detach().view(-1).mean().cpu().item())

        pred_xyxy, was_norm = maybe_norm_to_xyxy(pred, TRACK_W, TRACK_H)
        pred_xyxy = clamp_fix_xyxy(pred_xyxy, TRACK_W, TRACK_H)

        self.prev_img_t = img_t
        self.prev_box = pred_box_t.detach()  # 保持原坐标系

        return pred_xyxy, confv, was_norm


# =========================
# Main (sync 风格)
# =========================
def main():
    # 1) Events iterator (相机 or 文件)
    mv_iterator = EventsIterator(input_path=EVENT_INPUT, delta_t=10000)
    height, width = mv_iterator.get_size()

    # 文件模式：模拟实时（和 sync.py 一样）
    if not is_live_camera(EVENT_INPUT):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # 2) 模型
    detector = Detector(DET_CKPT_PATH)
    tracker = VideoTracker(TRACK_CKPT_PATH)

    # 3) 录制
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = None
    is_recording = False

    frame_idx = 0
    last_conf = 0.0

    # 4) Window
    with MTWindow(title="Live Det+Track (Metavision)", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_recording, video_writer
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

            # V: start/stop record
            if key == UIKeyEvent.KEY_V:
                if not is_recording:
                    video_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, SAVE_FPS, (width, height))
                    print("🎥 Recording started...")
                    is_recording = True
                else:
                    if video_writer is not None:
                        video_writer.release()
                    print(f"✅ Video saved to {OUT_VIDEO_PATH}")
                    is_recording = False

        window.set_keyboard_callback(keyboard_cb)

        # 5) CD frame generator（和 sync.py 一样）
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height, fps=SAVE_FPS, palette=ColorPalette.CoolWarm
        )

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal frame_idx, last_conf, is_recording, video_writer

            frame_bgr = cd_frame  # already BGR

            # -------- 重检策略 --------
            need_redetect = (frame_idx == 0)
            if REDETECT_EVERY > 0 and frame_idx > 0 and (frame_idx % REDETECT_EVERY == 0):
                need_redetect = True
            if frame_idx > 0 and last_conf < REDETECT_CONF_THRES:
                need_redetect = True

            if need_redetect:
                det_box_det = detector.infer_top1_box_xyxy_detspace(frame_bgr)
                if det_box_det is not None:
                    det_box_track = xyxy_rescale(det_box_det, (DET_IMG_SIZE, DET_IMG_SIZE), (TRACK_W, TRACK_H))
                    det_box_track = clamp_fix_xyxy(det_box_track, TRACK_W, TRACK_H)
                    tracker.reset()
                    tracker.init(frame_bgr, det_box_track)

            trk_box_track, confv, was_norm = tracker.step(frame_bgr)
            last_conf = confv

            vis = frame_bgr
            if trk_box_track is not None:
                # track space -> sensor space
                box_in_sensor = xyxy_rescale(trk_box_track, (TRACK_W, TRACK_H), (width, height))
                box_in_sensor = clamp_fix_xyxy(box_in_sensor, width, height)
                vis = draw_box(
                    vis, box_in_sensor, color=(0, 0, 255),
                    text=f"trk {frame_idx} conf={confv:.2f} norm={int(was_norm)}"
                )
            else:
                cv2.putText(vis, f"trk {frame_idx} (no box)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if PRINT_DEBUG_EVERY > 0 and (frame_idx % PRINT_DEBUG_EVERY == 0):
                print(f"[{frame_idx:05d}] conf={confv:.3f} was_norm={was_norm} box_track={trk_box_track}")

            # show
            window.show_async(vis)

            # record
            if is_recording and video_writer is not None:
                video_writer.write(vis)

            frame_idx += 1

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # 6) Process events
        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                break

        # cleanup
        if video_writer is not None:
            video_writer.release()
        print("Exit.")


if __name__ == "__main__":
    main()
