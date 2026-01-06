import torch.nn.functional as F

def warp_feature(feat, theta):
    """
    feat: [B, C, H, W]
    theta: [B, 2, 3]
    """
    B, C, H, W = feat.shape
    grid = F.affine_grid(theta, size=feat.size(), align_corners=False)
    warped = F.grid_sample(feat, grid, align_corners=False)
    return warped
