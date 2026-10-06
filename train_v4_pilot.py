from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_v3 import SuperResolutionDatasetV3
from model_v4 import SuperResolutionV4
from loss_v4 import V4Loss



# Configuration


DATASET = Path("dataset/DIV2K_train_HR")

CHECKPOINT_DIR = Path("checkpoints_v4_pilot")

CHECKPOINT_DIR.mkdir(exist_ok=True)

# Small pilot!
NUM_IMAGES = 200

PATCHES_PER_IMAGE = 5

BATCH_SIZE = 4

EPOCHS = 10

LEARNING_RATE = 0.0001

NUM_WORKERS = 0



# Device


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))



# Dataset


full_dataset = SuperResolutionDatasetV3(
    DATASET, crop_size=128, patches_per_image=PATCHES_PER_IMAGE
)

samples_per_image = PATCHES_PER_IMAGE

pilot_length = NUM_IMAGES * samples_per_image

pilot_indices = list(range(min(pilot_length, len(full_dataset))))

dataset = Subset(full_dataset, pilot_indices)

print("Pilot samples:", len(dataset))



# DataLoader


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)



# Model

model = SuperResolutionV4(num_blocks=12).to(DEVICE)


# Loss

criterion = V4Loss().to(DEVICE)


# Optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


# Mixed precision

scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")


# Training

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    total_pixel = 0.0
    total_perceptual = 0.0
    total_edge = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for lr, hr in progress:
        lr = lr.to(DEVICE, non_blocking=True)

        hr = hr.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        
        # Forward pass
        

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"
        ):
            sr = model(lr)

            
            sr = torch.clamp(sr, 0.0, 1.0)

            (loss, pixel, perceptual, edge) = criterion(sr, hr)

       
        # Backpropagation
       

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)

        scaler.update()

       
        # Statistics
        
        total_loss += loss.item()
        total_pixel += pixel.item()
        total_perceptual += perceptual.item()
        total_edge += edge.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(f"Total:      {total_loss / n:.6f}")

    print(f"Pixel:      {total_pixel / n:.6f}")

    print(f"Perceptual: {total_perceptual / n:.6f}")

    print(f"Edge:       {total_edge / n:.6f}")

    
    # Save

    checkpoint = CHECKPOINT_DIR / f"model_epoch_{epoch + 1}.pth"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint,
    )

    print("Saved:", checkpoint)


print("\nV4 pilot complete!")
