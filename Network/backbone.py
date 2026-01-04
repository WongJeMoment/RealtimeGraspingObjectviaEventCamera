import torch.nn as nn

class DWConv(nn.Module):
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, stride, 1, groups=c1)
        self.pw = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        return self.pw(self.dw(x))

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = DWConv(3, 16, 2)
        self.stage2 = DWConv(16, 32, 2)
        self.stage3 = DWConv(32, 64, 2)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x   # stride = 8
