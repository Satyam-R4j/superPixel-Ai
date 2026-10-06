from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset_v3 import SuperResolutionDatasetV3


from model_v3 import SuperResolutionV3


# Configuration

DATASET = Path("dataset/DIV2K_train_HR")

CHECKPOINT_DIR = Path("checkpoint_v3")

CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 100

LEARNING_RATE = 0.0002
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SuperResolutionDatasetV3(DATASET, crop_size=128, patches_per_image=10)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


model = SuperResolutionV3(num_blocks=12).to(DEVICE)

print("\nDevice:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# Loss
criterion = nn.L1Loss()

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning-rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# Tranining

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1} / {EPOCHS}")

    for lr, hr in progress:
        lr = lr.to(DEVICE, non_blocking=True)

        hr = hr.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        sr = model(lr)

        loss = criterion(sr, hr)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.5f}")

    scheduler.step()

    average_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch + 1}")

    print(f"Loss: {average_loss:.6f}")

    print("Learning rate:", scheduler.get_last_lr()[0])

    # Save checkpoints
    checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch + 1}.pth"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
        },
        checkpoint_path,
    )

    print("Saved:", checkpoint_path)

print("\nV3 training complete!")
