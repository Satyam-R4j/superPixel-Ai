import io
import random

import torch
from PIL import Image, ImageFilter
from torchvision import transforms


def degrade_image(hr_image):

    # 1. Random blur
    if random.random() < 0.5:
        radius = random.uniform(0.1, 1.5)

        hr_image = hr_image.filter(
            ImageFilter.GaussianBlur(radius)
        )

    # 2. Random resize
    scale = random.uniform(0.45, 0.65)
    width, height = hr_image.size
    
    lr_width = max(8, int(width * scale))
    lr_height = max(8, int(height * scale))

    lr_image = hr_image.resize(
        (lr_width, lr_height),
        Image.Resampling.BICUBIC
    )

    # 3. Random JPEG compression
    if random.random() < 0.5:

        buffer = io.BytesIO()

        quality = random.randint(30, 90)

        lr_image.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        lr_image = Image.open(
            buffer
        ).convert("RGB")

    # 4. Resize to target 64x64
    lr_image = lr_image.resize(
        (64, 64),
        Image.Resampling.BICUBIC
    )

    # 5. Add random noise

    lr_tensor = transforms.ToTensor()(
        lr_image
    )


    if random.random() < 0.5:

        noise_strength = random.uniform(
            0.0,
            0.03
        )

        noise = torch.randn_like(
            lr_tensor
        ) * noise_strength

        lr_tensor = lr_tensor + noise

        lr_tensor = torch.clamp(
            lr_tensor,
            0.0,
            1.0
        )

    return lr_tensor
