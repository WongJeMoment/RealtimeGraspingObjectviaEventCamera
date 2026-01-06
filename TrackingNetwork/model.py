import torch
import torch.nn as nn

from TrackingNetwork.MotionNet import MotionNet
from TrackingNetwork.backbone import Backbone
from TrackingNetwork.Warp import warp_feature
from TrackingNetwork.encoding import box_to_heatmap
from TrackingNetwork.head import TemporalDeltaBoxHead


class PMD_TSD_Box(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = MotionNet()
        self.backbone = Backbone()
        # 64(t) + 64(warp tm1) + 1(heatmap) = 129
        self.head = TemporalDeltaBoxHead(in_ch=64 * 2 + 1)

    def forward(self, img_t, img_tm1, box_tm1):
        """
        输入:
          img_t:    (B,3,H,W)
          img_tm1:  (B,3,H,W)
          box_tm1:  (B,4)  (假设xyxy，坐标基于输入图像尺寸)

        输出:
          pred_box_t: (B,4)
          conf:       (B,1) or (B,)
        """
        # 1) backbone features
        feat_t = self.backbone(img_t)      # (B,64,h,w)
        feat_tm1 = self.backbone(img_tm1)  # (B,64,h,w)

        # 2) motion / flow
        #    你的 MotionNet 可能是 motion(img_t, img_tm1) 或 motion(concat)
        try:
            flow = self.motion(img_t, img_tm1)
        except TypeError:
            # fallback: 如果 MotionNet forward 只接收拼接输入
            flow = self.motion(torch.cat([img_t, img_tm1], dim=1))

        # 3) warp tm1 feature to t
        feat_tm1_warp = warp_feature(feat_tm1, flow)

        # 4) box heatmap (1 channel) with same spatial size as features
        B, C, h, w = feat_t.shape

        # box_to_heatmap 的签名在不同实现里不一样：
        # 常见：box_to_heatmap(box, h, w) -> (B,1,h,w)
        # 如果你那边是 box_to_heatmap(box, (h,w))，就在这里改一下
        try:
            heat = box_to_heatmap(box_tm1, h, w)
        except TypeError:
            heat = box_to_heatmap(box_tm1, (h, w))

        # 保证 heat shape 正确
        if heat.dim() == 3:
            heat = heat.unsqueeze(1)  # (B,1,h,w)
        if heat.shape[-2:] != (h, w):
            heat = nn.functional.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)

        # 5) concat -> head
        x = torch.cat([feat_t, feat_tm1_warp, heat], dim=1)  # (B,129,h,w)

        out = self.head(x)

        # head 输出可能是 (delta, conf) 或 dict
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            delta_box, conf = out[0], out[1]
        elif isinstance(out, dict) and ("delta" in out) and ("conf" in out):
            delta_box, conf = out["delta"], out["conf"]
        else:
            raise RuntimeError(
                "TemporalDeltaBoxHead output format not recognized. "
                "Expected (delta_box, conf) or {'delta':..., 'conf':...}."
            )

        # 6) refine
        pred_box_t = box_tm1 + delta_box

        return pred_box_t, conf

    @torch.no_grad()
    def forward_single(self, img_t):
        """
        方案1首帧：
        - 用 img_tm1=img_t
        - box_tm1 用中心框初始化
        - 直接走 forward() 的完整时序路径（保证通道一致）
        """
        B, _, H, W = img_t.shape
        device = img_t.device
        dtype = img_t.dtype

        cx, cy = W / 2.0, H / 2.0
        bw, bh = W * 0.5, H * 0.5

        box_tm1 = torch.tensor(
            [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
            device=device, dtype=dtype
        ).unsqueeze(0).repeat(B, 1)

        img_tm1 = img_t.clone()
        return self.forward(img_t, img_tm1, box_tm1)
