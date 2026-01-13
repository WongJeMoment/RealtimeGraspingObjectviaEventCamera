# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple

import sys
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
EVENT_INPUT = ""  # "" => live camera; or "/path/to/file.raw" or camera serial

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

# =========================
# ✅ 关键：tracker box 坐标系（别猜）
# True: tracker 输出/输入为归一化 xyxy (0~1)
# False: tracker 输出/输入为像素 xyxy (0~W/H)
# =========================
TRACK_BOX_IS_NORMALIZED = True

# -------------------------
# 重检策略（不要 1）
# -------------------------
REDETECT_EVERY = 1          # 每 N 帧允许重检一次（0 关闭）
REDETECT_CONF_THRES = 0.65  # tracker conf 低于该值 => 允许重检
PRINT_DEBUG_EVERY = 10      # 打印频率（0 关闭）

# -------------------------
# 门控 + 稳框（不会锁死的版本）
# -------------------------
GATE_IOU_THRES = 0.05        # det 与 trk 的 IoU 门控（先放低，确保能动）
DET_MERGE_ALPHA = 0.45       # 接受 det 时：融合比例
EMA_ALPHA = 0.18             # 输出平滑
LIMIT_ONLY_WHEN_LOWCONF = True
LIMIT_CONF_THRES = 0.60

MAX_CENTER_STEP = 0.35       # 只在低 conf 时限制中心移动（比例按上一帧框大小）
MAX_SCALE_CHANGE = 0.45      # 只在低 conf 时限制尺度变化
MIN_WH = 8


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
        img = img[:, :, ::-1].copy()
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


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (aa + bb - inter + 1e-6))


def xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b.astype(np.float32)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_to_xyxy(v: np.ndarray) -> np.ndarray:
    cx, cy, w, h = v.astype(np.float32)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def limit_motion_and_scale(prev_xyxy: np.ndarray, cur_xyxy: np.ndarray) -> np.ndarray:
    """只做“轻量限位”：防止 det/track 抽风时突然飞走或变大。"""
    p = xyxy_to_cxcywh(prev_xyxy)
    c = xyxy_to_cxcywh(cur_xyxy)
    pcx, pcy, pw, ph = p
    ccx, ccy, cw, ch = c

    pw = max(pw, MIN_WH)
    ph = max(ph, MIN_WH)
    cw = max(cw, MIN_WH)
    ch = max(ch, MIN_WH)

    max_dx = MAX_CENTER_STEP * pw
    max_dy = MAX_CENTER_STEP * ph
    ccx = pcx + float(np.clip(ccx - pcx, -max_dx, max_dx))
    ccy = pcy + float(np.clip(ccy - pcy, -max_dy, max_dy))

    max_w = pw * (1.0 + MAX_SCALE_CHANGE)
    min_w = pw * (1.0 - MAX_SCALE_CHANGE)
    max_h = ph * (1.0 + MAX_SCALE_CHANGE)
    min_h = ph * (1.0 - MAX_SCALE_CHANGE)
    cw = float(np.clip(cw, min_w, max_w))
    ch = float(np.clip(ch, min_h, max_h))

    return cxcywh_to_xyxy(np.array([ccx, ccy, cw, ch], dtype=np.float32))


class BoxEMA:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state: Optional[np.ndarray] = None  # xyxy

    def reset(self):
        self.state = None

    def update(self, box_xyxy: np.ndarray) -> np.ndarray:
        box_xyxy = box_xyxy.astype(np.float32)
        if self.state is None:
            self.state = box_xyxy.copy()
            return box_xyxy
        self.state = (1.0 - self.alpha) * self.state + self.alpha * box_xyxy
        return self.state.copy()


def to_track_feed(box_xyxy_track: np.ndarray) -> np.ndarray:
    """把 track-space 的像素 xyxy 转成网络需要的输入格式"""
    if TRACK_BOX_IS_NORMALIZED:
        x1, y1, x2, y2 = box_xyxy_track.astype(np.float32)
        return np.array([x1 / TRACK_W, y1 / TRACK_H, x2 / TRACK_W, y2 / TRACK_H], np.float32)
    return box_xyxy_track.astype(np.float32)


def from_track_pred(pred4: np.ndarray) -> np.ndarray:
    """把网络输出 pred4 转成 track-space 像素 xyxy"""
    pred4 = pred4.astype(np.float32)
    if TRACK_BOX_IS_NORMALIZED:
        x1, y1, x2, y2 = pred4
        return np.array([x1 * TRACK_W, y1 * TRACK_H, x2 * TRACK_W, y2 * TRACK_H], np.float32)
    return pred4


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
        self.prev_box_feed: Optional[torch.Tensor] = None  # 网络输入格式（归一化或像素）
        self.prev_box_xyxy_track: Optional[np.ndarray] = None  # track-space 像素 xyxy

    @torch.no_grad()
    def init(self, frame_bgr: np.ndarray, init_box_xyxy_trackspace: np.ndarray):
        img_t = frame_to_tensor(frame_bgr, (TRACK_H, TRACK_W), bgr2rgb=True).to(DEVICE)
        init_box_xyxy_trackspace = clamp_fix_xyxy(init_box_xyxy_trackspace, TRACK_W, TRACK_H)

        feed = to_track_feed(init_box_xyxy_trackspace).reshape(1, 4)
        self.prev_img_t = img_t
        self.prev_box_feed = torch.from_numpy(feed).to(DEVICE)
        self.prev_box_xyxy_track = init_box_xyxy_trackspace.copy()

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray):
        if self.prev_img_t is None or self.prev_box_feed is None:
            return None, 0.0

        img_t = frame_to_tensor(frame_bgr, (TRACK_H, TRACK_W), bgr2rgb=True).to(DEVICE)

        pred_box_t, conf = self.model(img_t, self.prev_img_t, self.prev_box_feed)

        pred4 = pred_box_t.detach().view(-1).cpu().numpy().astype(np.float32)
        confv = float(conf.detach().view(-1).mean().cpu().item())

        pred_xyxy = from_track_pred(pred4)
        pred_xyxy = clamp_fix_xyxy(pred_xyxy, TRACK_W, TRACK_H)

        # 更新状态：喂回去用同一种坐标格式
        self.prev_img_t = img_t
        self.prev_box_feed = torch.from_numpy(to_track_feed(pred_xyxy).reshape(1, 4)).to(DEVICE)
        self.prev_box_xyxy_track = pred_xyxy.copy()

        return pred_xyxy, confv


# =========================
# Main
# =========================
def main():
    mv_iterator = EventsIterator(input_path=EVENT_INPUT, delta_t=10000)
    height, width = mv_iterator.get_size()

    if not is_live_camera(EVENT_INPUT):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    detector = Detector(DET_CKPT_PATH)
    tracker = VideoTracker(TRACK_CKPT_PATH)
    ema = BoxEMA(alpha=EMA_ALPHA)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = None
    is_recording = False

    frame_idx = 0
    last_conf = 0.0
    last_box_track: Optional[np.ndarray] = None

    with MTWindow(title="Live Det+Track (Metavision)", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_recording, video_writer
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

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

        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height, fps=SAVE_FPS, palette=ColorPalette.CoolWarm
        )

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal frame_idx, last_conf, last_box_track, is_recording, video_writer

            frame_bgr = cd_frame  # BGR

            # 1) 跟踪一步
            trk_box, confv = tracker.step(frame_bgr)
            last_conf = confv

            # 2) 是否允许重检
            allow_redetect = False
            if frame_idx == 0 or trk_box is None:
                allow_redetect = True
            if REDETECT_EVERY > 0 and frame_idx > 0 and (frame_idx % REDETECT_EVERY == 0):
                allow_redetect = True
            if frame_idx > 0 and confv < REDETECT_CONF_THRES:
                allow_redetect = True

            det_used = False

            # 3) 重检 + IoU 门控 + 融合
            if allow_redetect:
                det_box_det = detector.infer_top1_box_xyxy_detspace(frame_bgr)
                if det_box_det is not None:
                    det_box_track = xyxy_rescale(det_box_det, (DET_IMG_SIZE, DET_IMG_SIZE), (TRACK_W, TRACK_H))
                    det_box_track = clamp_fix_xyxy(det_box_track, TRACK_W, TRACK_H)

                    if trk_box is None:
                        tracker.reset()
                        tracker.init(frame_bgr, det_box_track)
                        ema.reset()
                        trk_box = det_box_track.copy()
                        det_used = True
                    else:
                        iou = iou_xyxy(trk_box, det_box_track)
                        if (iou >= GATE_IOU_THRES) or (confv < (REDETECT_CONF_THRES * 0.7)):
                            merged = (1.0 - DET_MERGE_ALPHA) * trk_box + DET_MERGE_ALPHA * det_box_track
                            merged = clamp_fix_xyxy(merged, TRACK_W, TRACK_H)

                            tracker.reset()
                            tracker.init(frame_bgr, merged)
                            ema.reset()
                            trk_box = merged.copy()
                            det_used = True

            # 4) 稳框（不会锁死）：仅低置信时限位
            vis = frame_bgr
            if trk_box is not None:
                if last_box_track is not None:
                    if (not LIMIT_ONLY_WHEN_LOWCONF) or (confv < LIMIT_CONF_THRES):
                        trk_box = limit_motion_and_scale(last_box_track, trk_box)
                        trk_box = clamp_fix_xyxy(trk_box, TRACK_W, TRACK_H)

                smooth = ema.update(trk_box)
                last_box_track = smooth.copy()

                # track -> sensor
                box_in_sensor = xyxy_rescale(smooth, (TRACK_W, TRACK_H), (width, height))
                box_in_sensor = clamp_fix_xyxy(box_in_sensor, width, height)

                vis = draw_box(
                    vis, box_in_sensor,
                    color=(0, 0, 255),
                    text=f"trk {frame_idx} conf={confv:.2f} det={int(det_used)} norm={int(TRACK_BOX_IS_NORMALIZED)}"
                )
            else:
                last_box_track = None
                ema.reset()
                cv2.putText(vis, f"trk {frame_idx} (no box)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if PRINT_DEBUG_EVERY > 0 and (frame_idx % PRINT_DEBUG_EVERY == 0):
                print(f"[{frame_idx:05d}] conf={confv:.3f} det_used={det_used} box_track={trk_box}")

            window.show_async(vis)

            if is_recording and video_writer is not None:
                video_writer.write(vis)

            frame_idx += 1

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)
            if window.should_close():
                break

        if video_writer is not None:
            video_writer.release()
        print("Exit.")


if __name__ == "__main__":
    main()
