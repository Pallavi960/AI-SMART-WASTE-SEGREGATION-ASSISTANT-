# ============================================================
# PHASE 2: DATA PREPROCESSING
# AI Smart Waste Segregation Assistant (SDG 12)
# Resize images, normalize, split into train/validation sets
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Configuration ──────────────────────────────────────────
DATASET_DIR  = "dataset"
IMG_SIZE     = (224, 224)   # Standard size for CNNs
BATCH_SIZE   = 32
VALIDATION_SPLIT = 0.2      # 80% train / 20% validation
SEED         = 42

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# ────────────────────────────────────────────────────────────


def create_data_generators():
    """
    Creates augmented training data generator and clean validation generator.
    Returns: train_gen, val_gen, class_indices
    """
    print("=" * 55)
    print("  PHASE 2: Data Preprocessing")
    print("=" * 55)

    # ── Training generator  (with augmentation) ─────────────
    # Augmentation makes the model more robust by creating
    # slightly modified copies of images during training.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,          # Normalize pixels 0→1
        rotation_range=20,          # Random rotation ±20°
        width_shift_range=0.1,      # Horizontal shift
        height_shift_range=0.1,     # Vertical shift
        shear_range=0.1,            # Shear transformation
        zoom_range=0.1,             # Random zoom
        horizontal_flip=True,       # Mirror images
        validation_split=VALIDATION_SPLIT
    )

    # ── Validation generator (NO augmentation) ──────────────
    # Validation images should be clean for honest evaluation.
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT
    )

    # ── Load training data ───────────────────────────────────
    print("\n📂 Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",   # One-hot encoding for 6 classes
        subset="training",
        seed=SEED,
        shuffle=True
    )

    # ── Load validation data ─────────────────────────────────
    print("📂 Loading validation data...")
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False
    )

    # ── Summary ──────────────────────────────────────────────
    print(f"\n✅ Training samples   : {train_gen.samples}")
    print(f"✅ Validation samples : {val_gen.samples}")
    print(f"✅ Classes detected   : {list(train_gen.class_indices.keys())}")
    print(f"✅ Image size         : {IMG_SIZE}")
    print(f"✅ Batch size         : {BATCH_SIZE}")

    return train_gen, val_gen, train_gen.class_indices


def visualize_samples(train_gen, n=9):
    """Show a grid of sample training images with their labels."""
    print("\n📊 Visualizing sample images...")

    images, labels = next(train_gen)     # Get one batch
    class_names = list(train_gen.class_indices.keys())

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle("Sample Training Images (after preprocessing)", fontsize=14)

    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(images[i])
            label_idx = np.argmax(labels[i])
            ax.set_title(class_names[label_idx], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=100)
    plt.show()
    print("✅ Sample grid saved as 'sample_images.png'")


# ── Run standalone ───────────────────────────────────────────
if __name__ == "__main__":
    train_gen, val_gen, class_indices = create_data_generators()
    visualize_samples(train_gen)
    print("\n✅ Preprocessing complete! Proceed to Phase 3.\n")
