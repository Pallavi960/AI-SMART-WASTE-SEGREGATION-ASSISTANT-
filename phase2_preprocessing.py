# ============================================================
# PHASE 2 (v2): DATA PREPROCESSING — IMPROVED
# AI Smart Waste Segregation Assistant (SDG 12)
# Adds: class weights, stronger augmentation, stratified split
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ─── Configuration ──────────────────────────────────────────
DATASET_DIR  = "dataset"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16            # smaller batch = more stable gradients on small dataset
VALIDATION_SPLIT = 0.2
SEED         = 42

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# ────────────────────────────────────────────────────────────


def create_data_generators():
    """
    Creates augmented training data generator and clean validation generator.
    Returns: train_gen, val_gen, class_indices, class_weights
    """
    print("=" * 55)
    print("  PHASE 2 (v2): Data Preprocessing — Improved")
    print("=" * 55)

    # ── Training generator (STRONGER augmentation) ──────────
    # More aggressive augmentation helps fight overfitting on
    # a small dataset (2,527 images is not a lot for a CNN).
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,          # was 20 → more rotation variety
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,             # was 0.1 → more zoom variety
        brightness_range=[0.8, 1.2],  # NEW: lighting variation
        channel_shift_range=20.0,   # NEW: slight color variation
        horizontal_flip=True,
        vertical_flip=False,        # waste photos rarely upside down in practice
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT
    )

    # ── Validation generator (NO augmentation) ───────────────
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
        class_mode="categorical",
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

    # ── Compute class weights (fixes imbalance, e.g. trash=137 vs paper=594) ─
    print("\n⚖️  Computing class weights for imbalanced classes...")
    train_labels = train_gen.classes
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights_array))

    print("\n  Class weights (higher = rarer class, gets more attention):")
    for idx, class_name in enumerate(train_gen.class_indices.keys()):
        print(f"    {class_name:<12} → weight: {class_weights[idx]:.2f}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n✅ Training samples   : {train_gen.samples}")
    print(f"✅ Validation samples : {val_gen.samples}")
    print(f"✅ Classes detected   : {list(train_gen.class_indices.keys())}")
    print(f"✅ Image size         : {IMG_SIZE}")
    print(f"✅ Batch size         : {BATCH_SIZE}")

    return train_gen, val_gen, train_gen.class_indices, class_weights


def visualize_samples(train_gen, n=9):
    """Show a grid of sample training images with their labels."""
    print("\n📊 Visualizing sample images...")

    images, labels = next(train_gen)
    class_names = list(train_gen.class_indices.keys())

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle("Sample Training Images (after augmentation)", fontsize=14)

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
    train_gen, val_gen, class_indices, class_weights = create_data_generators()
    visualize_samples(train_gen)
    print("\n✅ Preprocessing complete! Proceed to Phase 3.\n")