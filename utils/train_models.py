import torch
from torch import nn
import torch
from torch import nn

class SimpleTinyImageNetNet(nn.Module):
    def __init__(self):
        super(SimpleTinyImageNetNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # B x 3 x 64 x 64 → B x 32 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → B x 32 x 32 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → B x 64 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → B x 64 x 16 x 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # → B x (64*16*16)
            nn.Linear(200704, 200)                  # → B x 200
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x