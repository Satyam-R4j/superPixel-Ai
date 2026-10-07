from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model_v4 import SuperResolutionV4


# ============================================================
# Configuration
# ============================================================

INPUT_IMAGE = Path(
    "test_images/Pic.png"
)

MODEL_PATH = Path(
    "checkpoints_v4_pilot/model_epoch_10.pth"
)

OUTPUT_DIR = Path(
    "outputs/v4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Load model
# ============================================================

model = SuperResolutionV4(
    num_blocks=12
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("V4 model loaded.")


# ============================================================
# Load image
# ============================================================

image = Image.open(
    INPUT_IMAGE
).convert("RGB")

print(
    "Original size:",
    image.size
)


# ============================================================
# Controlled benchmark
# ============================================================

# Ground-truth image
hr = image.resize(
    (128, 128),
    Image.Resampling.LANCZOS
)

# Artificial low-resolution image
lr = hr.resize(
    (64, 64),
    Image.Resampling.BICUBIC
)

hr.save(
    OUTPUT_DIR /
    "01_original_128.png"
)

lr.save(
    OUTPUT_DIR /
    "02_low_resolution_64.png"
)


# ============================================================
# Inference
# ============================================================

to_tensor = transforms.ToTensor()

lr_tensor = to_tensor(
    lr
).unsqueeze(0).to(device)

with torch.no_grad():

    sr = model(
        lr_tensor
    )

sr = torch.clamp(
    sr,
    0,
    1
)

sr_image = transforms.ToPILImage()(
    sr.squeeze(0).cpu()
)

sr_image.save(
    OUTPUT_DIR /
    "03_v4_super_resolution.png"
)


print("\nV4 test complete.")

print(
    "Saved:",
    OUTPUT_DIR /
    "03_v4_super_resolution.png"
)