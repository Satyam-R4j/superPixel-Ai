
import torch
import torch.nn as nn


class SimpleSuperResolutionCNN(nn.Module):


    def __init__(self):
        super().__init__()

        # Feature extration
        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=64*4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(upscale_factor=2),

            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x):

        x = self.features(x)
        
        x = self.upsample(x)

        x = self.output(x)

        return x


