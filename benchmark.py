from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import ImprovedSuperResolutionCNN


# -------------------------
# Configuration
# -------------------------

INPUT = Path("test_images/Pic.jpeg")

MODEL = Path(
    "checkpoints_v2/model_epoch_75.pth"
)

OUTPUT = Path("outputs/benchmark")
OUTPUT.mkdir(parents=True, exist_ok=True)


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# -------------------------
# Load model
# -------------------------

model = ImprovedSuperResolutionCNN().to(device)

checkpoint = torch.load(
    MODEL,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -------------------------
# Load original
# -------------------------

image = Image.open(INPUT).convert("RGB")

print("Original:", image.size)


# -------------------------
# Create controlled images
# -------------------------

# Make a 128x128 HR image
hr = image.resize(
    (128, 128),
    Image.Resampling.LANCZOS
)

# Make a 64x64 LR image
lr = hr.resize(
    (64, 64),
    Image.Resampling.BICUBIC
)

hr.save(
    OUTPUT / "01_original_128.png"
)

lr.save(
    OUTPUT / "02_low_resolution_64.png"
)


# -------------------------
# Model inference
# -------------------------

to_tensor = transforms.ToTensor()

lr_tensor = to_tensor(
    lr
).unsqueeze(0).to(device)


with torch.no_grad():

    sr = model(
        lr_tensor
    )


# -------------------------
# Convert output
# -------------------------

sr = torch.clamp(
    sr,
    0,
    1
)

sr_image = transforms.ToPILImage()(
    sr.squeeze(0).cpu()
)

sr_image.save(
    OUTPUT / "03_ai_128.png"
)


print("\nCreated:")

print(
    OUTPUT / "01_original_128.png"
)

print(
    OUTPUT / "02_low_resolution_64.png"
)

print(
    OUTPUT / "03_ai_128.png"
)