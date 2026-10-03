import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __int__(self, channels=64):

        super.__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x + residual


class SuperResolutionV3(nn.Module):
    def __init__(self, num_blocks=12):

        super().__init__()
        self.head = nn.Conv2d(3, 64, 3, padding=1)

        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])

        self.body_conv = nn.Conv2d(64, 64, 3, padding=1)

        self.upsmaple = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(64, 3, 3, padding=1)

    
    def forward(self, x):
        features = self.head(x)

        body = self.body(features)

        body = self.body_conv(body)

        features = features + body
         
        output = self.tail(
            features
        )

        return output
