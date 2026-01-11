import torch
import torch.nn as nn

class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        diff = torch.abs(f1 - f2)

        out = self.decoder(diff)
        return out
