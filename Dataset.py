import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as TF

class SingleFrameDataset(Dataset):
    """
    self.data: list[str]  每条是图片路径
    label: 与图片同名的 txt，内容 YOLO 格式：cls cx cy w h（归一化）
    返回:
      {"img": float32 [3,H,W] in [0,1], "bbox": float32 [4] (cx,cy,w,h) norm}
    """
    def __init__(
        self,
        data_list,              # list[str] image paths
        label_dir=None,         # 如果 txt 和图片在同一目录，可传 None
        img_size=(256, 256),
        augment=True,
        p_flip=0.5,
        p_color=0.8,
        p_erase=0.25,
        p_geom=0.5,
        choose_box="first",     # "first" | "random" | "largest"
    ):
        super().__init__()
        self.data = data_list
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        self.p_flip = p_flip
        self.p_color = p_color
        self.p_erase = p_erase
        self.p_geom = p_geom

        self.choose_box = choose_box

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _clamp_bbox_norm(cx, cy, w, h):
        w = float(max(w, 1e-6))
        h = float(max(h, 1e-6))
        cx = float(min(max(cx, 0.0), 1.0))
        cy = float(min(max(cy, 0.0), 1.0))
        w = float(min(max(w, 0.0), 1.0))
        h = float(min(max(h, 0.0), 1.0))
        return cx, cy, w, h

    def _label_path_from_img(self, img_path: str) -> str:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if self.label_dir is None:
            # 和图片同目录
            return os.path.join(os.path.dirname(img_path), stem + ".txt")
        else:
            return os.path.join(self.label_dir, stem + ".txt")

    def _read_yolo_bboxes(self, label_path: str):
        """
        读取一个 txt 中所有 bbox:
        返回 list of (cls:int, cx,cy,w,h) 都是 float
        """
        if not os.path.exists(label_path):
            return []

        records = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                cx, cy, w, h = self._clamp_bbox_norm(cx, cy, w, h)
                records.append((cls, cx, cy, w, h))
        return records

    def _select_one_bbox(self, records):
        """
        单目标训练：从多框里选一个 bbox
        """
        if len(records) == 0:
            # 没标注就给一个极小框（避免崩，但训练可能没意义；你也可以改成 raise）
            return torch.tensor([0.5, 0.5, 1e-6, 1e-6], dtype=torch.float32)

        if self.choose_box == "random":
            cls, cx, cy, w, h = random.choice(records)
        elif self.choose_box == "largest":
            # 按面积选最大框
            cls, cx, cy, w, h = max(records, key=lambda r: r[3] * r[4])
        else:
            # 默认第一行
            cls, cx, cy, w, h = records[0]

        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    def _random_geom_weak(self, img, bbox):
        cx, cy, w, h = bbox.tolist()
        C, H, W = img.shape

        scale = random.uniform(0.9, 1.1)
        newH = max(1, int(round(H * scale)))
        newW = max(1, int(round(W * scale)))
        img2 = TF.resize(img, [newH, newW], antialias=True)

        cx2, cy2, w2, h2 = cx, cy, w, h

        padH = max(0, H - newH)
        padW = max(0, W - newW)
        pad_top = padH // 2
        pad_bottom = padH - pad_top
        pad_left = padW // 2
        pad_right = padW - pad_left

        if padH > 0 or padW > 0:
            img2 = TF.pad(img2, [pad_left, pad_top, pad_right, pad_bottom])

            canvasH = newH + padH
            canvasW = newW + padW

            x = cx2 * newW + pad_left
            y = cy2 * newH + pad_top
            bw = w2 * newW
            bh = h2 * newH

            cx2 = x / canvasW
            cy2 = y / canvasH
            w2 = bw / canvasW
            h2 = bh / canvasH

            newH, newW = canvasH, canvasW

        if newH > H:
            i = random.randint(0, newH - H)
        else:
            i = 0
        if newW > W:
            j = random.randint(0, newW - W)
        else:
            j = 0

        img3 = TF.crop(img2, i, j, H, W)

        x = cx2 * newW
        y = cy2 * newH
        bw = w2 * newW
        bh = h2 * newH

        x = x - j
        y = y - i

        cx3 = x / W
        cy3 = y / H
        w3 = bw / W
        h3 = bh / H

        cx3, cy3, w3, h3 = self._clamp_bbox_norm(cx3, cy3, w3, h3)
        return img3, torch.tensor([cx3, cy3, w3, h3], dtype=torch.float32)

    def _random_color(self, img):
        b = random.uniform(0.8, 1.2)
        c = random.uniform(0.8, 1.2)
        s = random.uniform(0.8, 1.2)
        h = random.uniform(-0.02, 0.02)

        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        img = TF.adjust_saturation(img, s)
        img = TF.adjust_hue(img, h)
        return img.clamp(0, 1)

    def _random_erase(self, img):
        C, H, W = img.shape
        area = H * W
        erase_area = random.uniform(0.02, 0.15) * area
        aspect = random.uniform(0.3, 3.3)

        hh = int(round((erase_area * aspect) ** 0.5))
        ww = int(round((erase_area / aspect) ** 0.5))
        if hh <= 0 or ww <= 0:
            return img

        y = random.randint(0, max(0, H - hh))
        x = random.randint(0, max(0, W - ww))

        img = img.clone()
        img[:, y:y+hh, x:x+ww] = 0.0
        return img

    def __getitem__(self, idx):
        img_path = self.data[idx]  # ✅ 现在 item 就是图片路径字符串

        # 读对应的 yolo txt
        label_path = self._label_path_from_img(img_path)
        records = self._read_yolo_bboxes(label_path)
        bbox = self._select_one_bbox(records)  # float32 [4] norm

        # 读图：uint8 [C,H,W]
        img = read_image(img_path)[:3]
        img = img.float() / 255.0

        # resize 到固定训练尺寸：bbox 是归一化，resize 不需要改
        img = TF.resize(img, list(self.img_size), antialias=True)

        if self.augment:
            # 1) 水平翻转（同步更新 bbox）
            if random.random() < self.p_flip:
                img = TF.hflip(img)
                cx, cy, w, h = bbox.tolist()
                cx = 1.0 - cx
                bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)

            # 2) 颜色增强（不改 bbox）
            if random.random() < self.p_color:
                img = self._random_color(img)

            # 3) 弱几何增强（同步更新 bbox）
            if random.random() < self.p_geom:
                img, bbox = self._random_geom_weak(img, bbox)

            # 4) 随机擦除（不改 bbox）
            if random.random() < self.p_erase:
                img = self._random_erase(img)

        return {"img": img, "bbox": bbox}
