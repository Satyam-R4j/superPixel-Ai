import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16, VGG16_Weights


# VGG perceptual network


class VGGPerceptualLoss(nn.Module):
    def __init__(self):

        super().__init__()

        weights = VGG16_Weights.DEFAULT

        vgg = vgg16(weights=weights)

        self.features = nn.Sequential(*list(vgg.features.children())[:16])

        self.features.eval()

        for parameter in self.features.parameters():
            parameter.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, sr, hr):

        sr = self.normalize(sr)
        hr = self.normalize(hr)

        sr_features = self.features(sr)
        hr_features = self.features(hr)

        return F.l1_loss(sr_features, hr_features)


# Edge loss


class EdgeLoss(nn.Module):
    def __init__(self):

        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)

        self.register_buffer("sobel_y", sobel_y)

    def edge_map(self, image):

        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

        gx = F.conv2d(gray, self.sobel_x, padding=1)

        gy = F.conv2d(gray, self.sobel_y, padding=1)

        edges = torch.sqrt(gx**2 + gy**2 + 1e-6)

        return edges

    def forward(self, sr, hr):

        sr_edges = self.edge_map(sr)
        hr_edges = self.edge_map(hr)

        return F.l1_loss(sr_edges, hr_edges)


# Combined V4 loss


class V4Loss(nn.Module):
    def __init__(self):

        super().__init__()

        self.pixel_loss = nn.L1Loss()

        self.perceptual_loss = VGGPerceptualLoss()

        self.edge_loss = EdgeLoss()

    def forward(self, sr, hr):

        pixel = self.pixel_loss(sr, hr)

        perceptual = self.perceptual_loss(sr, hr)

        edge = self.edge_loss(sr, hr)

        total = 1.0 * pixel + 0.1 * perceptual + 0.05 * edge

        return (total, pixel, perceptual, edge)
