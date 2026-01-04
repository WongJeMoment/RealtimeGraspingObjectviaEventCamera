import torch
import torch.nn.functional as F

def box_to_heatmap(box, feat_h, feat_w, sigma=2.0):
    """
    box: [B, 4] (cx, cy, w, h) normalized [0,1]
    return: [B, 1, H, W]
    """
    B = box.size(0)
    cx, cy = box[:, 0], box[:, 1]

    xs = torch.linspace(0, 1, feat_w, device=box.device)
    ys = torch.linspace(0, 1, feat_h, device=box.device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    heatmap = torch.exp(-((xx - cx[:, None, None]) ** 2 +
                           (yy - cy[:, None, None]) ** 2) / (2 * sigma ** 2))
    return heatmap.unsqueeze(1)
