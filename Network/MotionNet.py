import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionNet(nn.Module):
    """
    Predict global affine motion between I_{t-1} and I_t
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch * 2, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, 6)  # affine: 2x3

        # initialize as identity
        nn.init.zeros_(self.fc.weight)
        self.fc.bias.data = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)

    def forward(self, img_t, img_tm1):
        x = torch.cat([img_t, img_tm1], dim=1)
        feat = self.encoder(x).flatten(1)
        theta = self.fc(feat).view(-1, 2, 3)
        return theta
