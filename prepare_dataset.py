from pathlib import Path
from PIL import Image
import random

# Configuration

SOURCE_DIR = Path("dataset/DIV2K_train_HR")
OUTPUT_DIR = Path("dataset/sr_pairs")

NUM_PATCHES_PER_IMAGE = 10
HR_SIZE = 128
LR_SIZE = 64

TRAIN_RATIO = 0.9

random.seed(42)


# Create directories

train_hr = OUTPUT_DIR / "train" / "hr"
train_lr = OUTPUT_DIR / "train" / "lr"

val_hr = OUTPUT_DIR / "val" / "hr"
val_lr = OUTPUT_DIR / "val" / "lr"

for directory in [train_hr, train_lr, val_hr, val_lr]:
    directory.mkdir(parents=True, exist_ok=True)


# Find images

images = sorted(
    list(SOURCE_DIR.glob("*.png"))
    + list(SOURCE_DIR.glob("*.jpg"))
    + list(SOURCE_DIR.glob("*.jpeg"))
)

print(f"Found {len(images)} images.")

if len(images) == 0:
    raise RuntimeError(
        f"No images found in {SOURCE_DIR}"
    )


# Train / validation split

random.shuffle(images)

split_index = int(len(images) * TRAIN_RATIO)

train_images = images[:split_index]
val_images = images[split_index:]

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")


# Generate pairs

def create_pairs(images, hr_dir, lr_dir, prefix):

    pair_number = 0

    for image_path in images:

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        width, height = image.size

        # Image must be large enough for our crop
        if width < HR_SIZE or height < HR_SIZE:
            continue

        for _ in range(NUM_PATCHES_PER_IMAGE):

            # Random crop location
            x = random.randint(0, width - HR_SIZE)
            y = random.randint(0, height - HR_SIZE)

            hr_patch = image.crop(
                (x, y, x + HR_SIZE, y + HR_SIZE)
            )

            # Create LR image
            lr_patch = hr_patch.resize(
                (LR_SIZE, LR_SIZE),
                Image.Resampling.BICUBIC
            )

            filename = f"{prefix}_{pair_number:06d}.png"

            hr_patch.save(hr_dir / filename)
            lr_patch.save(lr_dir / filename)

            pair_number += 1

    return pair_number


# Generate training data

print("\nCreating training pairs...")

train_count = create_pairs(
    train_images,
    train_hr,
    train_lr,
    "train"
)

print(f"Created {train_count} training pairs.")


# Generate validation data

print("\nCreating validation pairs...")

val_count = create_pairs(
    val_images,
    val_hr,
    val_lr,
    "val"
)

print(f"Created {val_count} validation pairs.")


print("\nDataset preparation complete!")

print(f"\nTraining HR: {train_hr}")
print(f"Training LR: {train_lr}")
print(f"Validation HR: {val_hr}")
print(f"Validation LR: {val_lr}")