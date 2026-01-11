import torch
import torch.nn as nn
from unet_blocks import DoubleConv, Down, Up

class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Decoder
        # NOTE: input channels = encoder + skip
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)

        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x1, x2):
        # Encoder (shared weights)
        x1_1 = self.inc(x1)
        x1_2 = self.down1(x1_1)
        x1_3 = self.down2(x1_2)
        x1_4 = self.down3(x1_3)

        x2_1 = self.inc(x2)
        x2_2 = self.down1(x2_1)
        x2_3 = self.down2(x2_2)
        x2_4 = self.down3(x2_3)

        # Feature differences
        d1 = torch.abs(x1_1 - x2_1)  # 64
        d2 = torch.abs(x1_2 - x2_2)  # 128
        d3 = torch.abs(x1_3 - x2_3)  # 256
        d4 = torch.abs(x1_4 - x2_4)  # 512

        # Decoder
        x = self.up1(d4, d3)  # 512 + 256 → 256
        x = self.up2(x, d2)   # 256 + 128 → 128
        x = self.up3(x, d1)   # 128 + 64  → 64

        return torch.sigmoid(self.outc(x))