import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as TF

class SingleFrameDataset(Dataset):
    """
    假设 self.data 每条是:
      {"img_path": "...", "bbox": [cx, cy, w, h]}  # bbox 归一化到[0,1]
    img 返回 float32, 范围 [0,1], shape [3,H,W]
    bbox 返回 float32, shape [4] (cx,cy,w,h), 仍然归一化
    """
    def __init__(
        self,
        data_list,
        img_size=(256, 256),
        augment=True,
        p_flip=0.5,
        p_color=0.8,
        p_erase=0.25,
        p_geom=0.5,
    ):
        super().__init__()
        self.data = data_list
        self.img_size = img_size
        self.augment = augment

        self.p_flip = p_flip
        self.p_color = p_color
        self.p_erase = p_erase
        self.p_geom = p_geom

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _clamp_bbox_norm(cx, cy, w, h):
        # 防止异常值：w/h 最小给个下限
        w = float(max(w, 1e-6))
        h = float(max(h, 1e-6))
        cx = float(min(max(cx, 0.0), 1.0))
        cy = float(min(max(cy, 0.0), 1.0))
        w = float(min(max(w, 0.0), 1.0))
        h = float(min(max(h, 0.0), 1.0))
        return cx, cy, w, h

    def _random_geom_weak(self, img, bbox):
        """
        弱几何增强：随机缩放 + 平移（不旋转），并同步更新 bbox。
        做法：先 pad 再 crop 到原大小，等价于缩放/平移。
        """
        cx, cy, w, h = bbox.tolist()
        C, H, W = img.shape

        # 随机缩放比例（越接近1越弱）
        scale = random.uniform(0.9, 1.1)

        # 先把图 resize 到 scaled size
        newH = max(1, int(round(H * scale)))
        newW = max(1, int(round(W * scale)))
        img2 = TF.resize(img, [newH, newW], antialias=True)

        # 计算 bbox 在缩放后仍然是归一化坐标，因此：
        # 如果只是resize，归一化坐标不变（因为相对位置不变）
        cx2, cy2, w2, h2 = cx, cy, w, h

        # 为了能 crop 回 (H,W)，需要 pad 或 crop
        # 我们统一 pad 到至少(H,W)，再随机 crop
        padH = max(0, H - newH)
        padW = max(0, W - newW)
        # pad 分到上下左右
        pad_top = padH // 2
        pad_bottom = padH - pad_top
        pad_left = padW // 2
        pad_right = padW - pad_left

        if padH > 0 or padW > 0:
            img2 = TF.pad(img2, [pad_left, pad_top, pad_right, pad_bottom])

            # pad 会改变归一化坐标：原图内容被放到更大画布中
            canvasH = newH + padH
            canvasW = newW + padW

            # bbox 的中心点在 canvas 上的位置（以像素计）
            x = cx2 * newW + pad_left
            y = cy2 * newH + pad_top
            bw = w2 * newW
            bh = h2 * newH

            # 转回归一化（相对于 canvas）
            cx2 = x / canvasW
            cy2 = y / canvasH
            w2 = bw / canvasW
            h2 = bh / canvasH

            newH, newW = canvasH, canvasW

        # 随机 crop 到 (H,W)
        # crop 左上角 (i,j)
        if newH > H:
            i = random.randint(0, newH - H)
        else:
            i = 0
        if newW > W:
            j = random.randint(0, newW - W)
        else:
            j = 0

        img3 = TF.crop(img2, i, j, H, W)

        # crop 会让 bbox 相对坐标发生平移：从 canvas 坐标转到 crop 坐标
        # 先转像素，再减去 crop 偏移，再归一化到 (H,W)
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
        # img: float [0,1]
        # 颜色抖动（手写简单版，避免引入 Compose）
        # brightness/contrast/saturation/hue
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
        # 随机擦除：不改 bbox（当作遮挡增强）
        C, H, W = img.shape
        area = H * W
        erase_area = random.uniform(0.02, 0.15) * area
        aspect = random.uniform(0.3, 3.3)

        h = int(round((erase_area * aspect) ** 0.5))
        w = int(round((erase_area / aspect) ** 0.5))
        if h <= 0 or w <= 0:
            return img

        y = random.randint(0, max(0, H - h))
        x = random.randint(0, max(0, W - w))

        img = img.clone()
        img[:, y:y+h, x:x+w] = 0.0
        return img

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["img_path"]
        bbox = torch.tensor(item["bbox"], dtype=torch.float32)  # (cx,cy,w,h) norm

        # 读图：uint8 [C,H,W]
        img = read_image(img_path)[:3]  # 防止有 alpha
        img = img.float() / 255.0

        # resize 到固定训练尺寸：bbox 归一化则不需要改
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
