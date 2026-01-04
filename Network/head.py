import torch
import torch.nn as nn

class TemporalDeltaBoxHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 5)  # Δcx, Δcy, Δw, Δh, conf

    def forward(self, feat):
        x = self.conv(feat)
        x = self.pool(x).flatten(1)
        out = self.fc(x)
        delta_box = out[:, :4]
        conf = torch.sigmoid(out[:, 4])
        return delta_box, conf
