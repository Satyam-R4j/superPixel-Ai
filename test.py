from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


from model import SimpleSuperResolutionCNN


# configuration

INPUT_IMAGE = Path("test_images/Pic.jpg")

MODEL_PATH = Path(
    "checkpoints/model_epoch_20.pth"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using: ", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# Load model
model = SimpleSuperResolutionCNN().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()



# Load image

image = Image.open(
    INPUT_IMAGE
).convert("RGB")


print(
    "Original size:",
    image.size
)


# Create low-resolution image

width, height = image.size
width = (width // 2) * 2
height = (height // 2) * 2
image = image.crop((0, 0, width, height))
lr_size = (width // 2, height // 2)

lr_image = image.resize(
    lr_size,
    Image.Resampling.BICUBIC
)


lr_image.save(
    OUTPUT_DIR / "low_resolution.png"
)

# Convert image to tensor

to_tensor = transforms.ToTensor()

lr_tensor = to_tensor(
    lr_image
).unsqueeze(0)

lr_tensor = lr_tensor.to(device)


# Super-resolution

with torch.no_grad():

    sr_tensor = model(
        lr_tensor
    )


# Convert tensor back to image

sr_tensor = sr_tensor.squeeze(0)

sr_tensor = torch.clamp(
    sr_tensor,
    0,
    1
)

to_image = transforms.ToPILImage()

sr_image = to_image(
    sr_tensor.cpu()
)

# Save results

sr_image.save(
    OUTPUT_DIR / "super_resolved.png"
)

image.save(
    OUTPUT_DIR / "original.png"
)

print("\nDone!")

print(
    "Low resolution: ",
    OUTPUT_DIR / "low_resolution.png"
)

print(
    "Super resolution:",
    OUTPUT_DIR / "super_resolved.png"
)