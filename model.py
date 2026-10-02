
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):


    def __init__(self, channels = 64):
        super().__init__()

        # Feature extration
        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )           
        )

    def forward(self, x):
        return x + self.block(x)

       
class ImprovedSuperResolutionCNN(nn.Module):

    def __init__(self, num_blocks=8):
        super().__init__()

        self.input = nn.Conv2d(
            3,
            64,
            kernel_size = 3,
            padding=1
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(64)
                for _ in range(num_blocks)
            ]
        )

        self.refine = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )


        self.upsample = nn.Sequential(
            nn.Conv2d(
                64,
                64 * 4,
                kernel_size=3,
                padding = 1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(
            64,
            3,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x):

        features = self.input(x)

        residual = self.residual_blocks(features)

        residual = self.refine(residual)

        features = features + residual

        features = self.upsample(features)

        output = self.output(features)

        return output



