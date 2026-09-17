from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import SimpleSuperResolutionCNN



# Configuration
TRAIN_LR_DIR =  Path("dataset/sr_pairs/train/lr")
TRAIN_HR_DIR = Path("dataset/sr_pairs/train/hr")


VAL_LR_DIR = Path("dataset/sr_pairs/val/lr")
VAL_HR_DIR = Path("dataset/sr_pairs/val/hr")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0002

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Dataset

class SuperResolutionDataset(Dataset):

    def __init__(self, lr_dir, hr_dir):

        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)

        self.images = sorted(self.lr_dir.glob("*.png"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        lr_path = self.images[index]

        hr_path = self.hr_dir / lr_path.name

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        lr_tensor = self.to_tensor(lr_image)
        hr_tensor = self.to_tensor(hr_image)

        return lr_tensor, hr_tensor


# Create datsets

train_dataset = SuperResolutionDataset(
    TRAIN_LR_DIR,
    TRAIN_HR_DIR
)

val_dataset = SuperResolutionDataset(
    VAL_LR_DIR,
    VAL_HR_DIR
)

print("Training samples: " ,len(train_dataset))
print("Validation samples: " ,len(val_dataset))


# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers= NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# Model

model = SimpleSuperResolutionCNN().to(DEVICE)

print("\n Using device: ", DEVICE)

if DEVICE.type == "cuda":
    print("GPU: ", torch.cuda.get_device_name(0))


# Loss + Optimizer

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LEARNING_RATE
)

# Training

for epoch in range(NUM_EPOCHS):

    model.train()

    train_loss = 0.0

    progress = tqdm(
        train_loader,
        desc = f"Epoch {epoch + 1} / {NUM_EPOCHS}"
    )


    for lr_images, hr_images in progress:

        lr_images = lr_images.to(
            DEVICE,
            non_blocking = True
        )

        hr_images = hr_images.to(
            DEVICE,
            non_blocking= True
        )

        optimizer.zero_grad()

        sr_images = model(lr_images)

        loss = criterion(
            sr_images,
            hr_images
        )

        
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        progress.set_postfix(
            loss = f"{loss.item():.5f}"
        )
    
    train_loss /= len(train_loader)

    # Validation

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for lr_images, hr_images in val_loader:

            lr_images  = lr_images.to(DEVICE)
            hr_images = hr_images.to(DEVICE)

            sr_images = model(lr_images)

            loss = criterion(
                sr_images,
                hr_images
            )

            val_loss += loss.item()
        
        val_loss /= len(val_loader)

        print(f"\n Epoch {epoch + 1}/{NUM_EPOCHS}")

        print(f"Training Loss: {train_loss:.6f}")

        print(f"Validation Loss: {val_loss:.6f}")


    # Save checkpoint

    checkpoint_path = (
        CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        checkpoint_path
    )

    print(f"Saved: {checkpoint_path}")

print("\nTraining completed!")