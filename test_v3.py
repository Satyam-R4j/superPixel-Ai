from pathlib import Path


import torch
from PIL import Image
from torchvision import transforms

from model_v3 import SuperResolutionV3


# Configuration

INPUT_IMAGE = Path("test_images/Pic.png")

MODEL_PATH = Path("checkpoint_v3/model_epoch_100.pth")

OUTPUT_DIR = Path("outputs/v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

if device.type == "cuda":
    print("GPU: ", torch.cuda.get_device_name(0))


# Load Model

model = SuperResolutionV3(num_blocks=12).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("V3 model loaded.")

# Load image

image = Image.open(INPUT_IMAGE).convert("RGB")

print("Original size: ", image.size)

# Make controlled test
hr = image.resize((128, 128), Image.Resampling.LANCZOS)

lr = hr.resize((64, 64), Image.Resampling.BICUBIC)


hr.save(OUTPUT_DIR / "01_original_128.png")

lr.save(OUTPUT_DIR / "02_low_resolution_64.png")

# Interference

to_tensor = transforms.ToTensor()

lr_tensor = to_tensor(lr).unsqueeze(0).to(device)

with torch.no_grad():
    sr = model(lr_tensor)

# Convert output
sr = torch.clamp(sr, 0, 1)

sr_image = transforms.ToPILImage()(sr.squeeze(0).cpu())

sr_image.save(OUTPUT_DIR / "03_v3_super_resolution.png")

print("\nFinished!")

print("Original: ", OUTPUT_DIR / "01_original_128.png")

print("Low resolution: ", OUTPUT_DIR / "02_low_resolution_64.png")

print("V3: ", OUTPUT_DIR / "03_v3_super_resolution.png")
