from benchmark import lr_tensor
from pathlib import Path

import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from degradation import degrade_image


class SuperResolutionDatasetV3(Dataset):
    def __init__(self, image_dir, crop_size=128, pathches_per_image=10):

        self.image_dir = Path(image_dir)
        self.images = sorted(
            list(self.image_dir.glob(("*.png")))
            + list(self.image_dir.glob(("*.jpg")))
            + list(self.image_dir.glob(("*.jpeg")))
        )

        self.crop_size = crop_size
        self.patches_per_image = pathches_per_image

        self.to_tensor = transforms.ToTensor()

        print(f"Found {len(self.images)} HR images.")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, index):

        image_index = index // self.patches_per_image

        image_path = self.images(image_index)

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        if width < self.crop_size or height < self.crop_size:
            return self.__getitem__((index + 1) % len(self))

        x = random.randint(0, width - self.crop_size)

        y = random.randint(0, height - self.crop_size)

        hr = image.crop((x, y, x + self.crop_size, y + self.crop_size))

        if random.random() < 0.5:
            hr = hr.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
        
        if random.random() < 0.5:
            hr = hr.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
        
        if random.random() < 0.5:
            hr = hr.rotate(
                random.choice([
                    90, 180, 270
                ])
            )

        hr_tensor = self.to_tensor(hr)

        lr_tensor = degrade_image(hr)

        return lr_tensor, hr_tensor
