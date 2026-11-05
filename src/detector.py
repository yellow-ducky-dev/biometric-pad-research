"""
A small CNN binary classifier: real fingerprint vs. GAN-synthesized fingerprint.

This stands in for a Presentation Attack Detection (PAD) / liveness-style
system. It is trained from scratch on your data, same as the GAN — not a
pretrained ImageNet model repurposed with hand-waved semantics.
"""
import torch.nn as nn


class PADClassifier(nn.Module):
    def __init__(self, nc=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, 32, 3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).view(-1)
