import torch
import torch.nn as nn
from motion_net import MotionNet
from backbone import Backbone
from warp import warp_feature
from box_encoding import box_to_heatmap
from head import TemporalDeltaBoxHead

class PMD_TSD_Box(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = MotionNet()
        self.backbone = Backbone()
        self.head = TemporalDeltaBoxHead(in_ch=64*2 + 1)

    def forward(self, img_t, img_tm1, box_tm1):
        theta = self.motion(img_t, img_tm1)

        feat_t = self.backbone(img_t)
        feat_tm1 = self.backbone(img_tm1)
        feat_tm1 = warp_feature(feat_tm1, theta)

        B, C, H, W = feat_t.shape
        box_map = box_to_heatmap(box_tm1, H, W)

        feat = torch.cat([feat_t, feat_tm1, box_map], dim=1)

        delta_box, conf = self.head(feat)
        box_t = box_tm1 + delta_box

        return box_t, conf
