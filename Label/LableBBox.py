import os
import glob
import cv2
import numpy as np

# ========= 配置区 =========
IMG_DIR = "/home/wangzhe/2026/IROS/Dataset/Img/apple2"      # 图片目录
LBL_DIR = "/home/wangzhe/2026/IROS/Dataset/Label/apple2"      # 输出标签目录（YOLO格式txt）
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

WINDOW_NAME = "YOLO Label Tool"
NC = 20  # 类别数（仅用于提示，不做强约束）
# =========================

os.makedirs(LBL_DIR, exist_ok=True)


def list_images(img_dir):
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        imgs.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
    return sorted(imgs)


def img_to_label_path(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(LBL_DIR, base + ".txt")


def load_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return []
    lines = open(label_path, "r", encoding="utf-8").read().strip().splitlines()
    labels = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        labels.append((int(float(cls)), float(xc), float(yc), float(w), float(h)))
    return labels


def save_yolo_labels(label_path, labels):
    # labels: list of (cls, xc, yc, w, h) normalized
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def norm_xyxy_to_yolo(x1, y1, x2, y2, W, H):
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = np.clip(x1, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    x2 = np.clip(x2, 0, W - 1)
    y2 = np.clip(y2, 0, H - 1)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / W, yc / H, bw / W, bh / H


def yolo_to_abs_xyxy(xc, yc, w, h, W, H):
    xc *= W
    yc *= H
    w *= W
    h *= H
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return int(x1), int(y1), int(x2), int(y2)


class LabelTool:
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.i = 0

        self.img = None
        self.disp = None
        self.H = 0
        self.W = 0

        self.labels = []   # current labels (cls, xc,yc,w,h)
        self.label_path = ""

        # drawing state
        self.drawing = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0

        # last class id (for fast labeling)
        self.last_cls = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.load_image()

    def load_image(self):
        img_path = self.img_paths[self.i]
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise RuntimeError(f"Failed to read: {img_path}")
        self.H, self.W = self.img.shape[:2]

        self.label_path = img_to_label_path(img_path)
        self.labels = load_yolo_labels(self.label_path)

        self.redraw()

    def redraw(self):
        self.disp = self.img.copy()

        # draw existing boxes
        for idx, (cls, xc, yc, w, h) in enumerate(self.labels):
            x1, y1, x2, y2 = yolo_to_abs_xyxy(xc, yc, w, h, self.W, self.H)
            cv2.rectangle(self.disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.disp, f"{idx}:{cls}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # draw current dragging box
        if self.drawing:
            cv2.rectangle(self.disp, (self.x0, self.y0), (self.x1, self.y1), (0, 0, 255), 2)

        # overlay help text
        info = [
            f"[{self.i+1}/{len(self.img_paths)}] last_cls={self.last_cls} (NC={NC})",
            "Mouse: Drag LMB to draw box",
            "Keys: n-next  p-prev  s-save  u-undo  c-clear  d-del_last  q/esc-quit",
            "      0-9: set last_cls quickly,  i: input cls id",
        ]
        y = 25
        for line in info:
            cv2.putText(self.disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 25

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
            self.x1, self.y1 = x, y
            self.redraw()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x1, self.y1 = x, y
            self.redraw()

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.x1, self.y1 = x, y

            # ignore too small
            if abs(self.x1 - self.x0) < 3 or abs(self.y1 - self.y0) < 3:
                self.redraw()
                return

            xc, yc, w, h = norm_xyxy_to_yolo(self.x0, self.y0, self.x1, self.y1, self.W, self.H)

            cls = self.last_cls
            # add label
            self.labels.append((int(cls), float(xc), float(yc), float(w), float(h)))
            self.auto_save()
            self.redraw()

    def auto_save(self):
        save_yolo_labels(self.label_path, self.labels)

    def save(self):
        save_yolo_labels(self.label_path, self.labels)
        print("Saved:", self.label_path)

    def next_img(self):
        self.save()
        if self.i < len(self.img_paths) - 1:
            self.i += 1
            self.load_image()

    def prev_img(self):
        self.save()
        if self.i > 0:
            self.i -= 1
            self.load_image()

    def undo(self):
        if len(self.labels) > 0:
            self.labels.pop()
            self.auto_save()
            self.redraw()

    def clear(self):
        self.labels = []
        self.auto_save()
        self.redraw()

    def del_last(self):
        # alias of undo, kept for clarity
        self.undo()

    def input_cls(self):
        try:
            s = input(f"Input class id (0..{NC-1}) current={self.last_cls}: ").strip()
            if s == "":
                return
            v = int(s)
            self.last_cls = v
            self.redraw()
        except Exception as e:
            print("Invalid input:", e)

    def run(self):
        while True:
            cv2.imshow(WINDOW_NAME, self.disp)
            k = cv2.waitKey(20) & 0xFF

            if k in [27, ord('q')]:   # esc/q
                self.save()
                break
            elif k == ord('n'):
                self.next_img()
            elif k == ord('p'):
                self.prev_img()
            elif k == ord('s'):
                self.save()
            elif k == ord('u'):
                self.undo()
            elif k == ord('c'):
                self.clear()
            elif k == ord('d'):
                self.del_last()
            elif k == ord('i'):
                self.input_cls()
            elif ord('0') <= k <= ord('9'):
                self.last_cls = int(chr(k))
                self.redraw()

        cv2.destroyAllWindows()


def main():
    img_paths = list_images(IMG_DIR)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in: {IMG_DIR}")
    tool = LabelTool(img_paths)
    tool.run()


if __name__ == "__main__":
    main()