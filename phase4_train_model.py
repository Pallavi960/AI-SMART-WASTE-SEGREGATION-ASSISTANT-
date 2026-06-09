# ============================================================
# PHASE 4: TRAIN THE MODEL
# AI Smart Waste Segregation Assistant (SDG 12)
# Full training pipeline with fine-tuning
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import our phases
from phase2_preprocessing import create_data_generators, BATCH_SIZE
from phase3_build_model import (
    build_model_transfer,
    build_model_scratch,
    compile_model,
    print_model_summary
)

# ─── Configuration ──────────────────────────────────────────
EPOCHS_PHASE1   = 15    # Train top layers only
EPOCHS_PHASE2   = 10    # Fine-tune base model
MODEL_SAVE_PATH = "waste_model.h5"
USE_TRANSFER    = True  # True = MobileNetV2, False = scratch CNN
# ────────────────────────────────────────────────────────────


def get_callbacks():
    """
    Callbacks control training behavior:
    - EarlyStopping:   stop if no improvement
    - ModelCheckpoint: save the best model automatically
    - ReduceLROnPlateau: reduce LR when stuck
    """
    callbacks = [
        # Stop early if val_accuracy doesn't improve for 5 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Always save the BEST model (by val_accuracy)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Cut learning rate if stuck for 3 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def plot_history(history, phase_name="Phase 1"):
    """Plot training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Results — {phase_name}", fontsize=14)

    # ── Accuracy plot ────────────────────────────────────────
    ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="#2ecc71")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#e74c3c")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # ── Loss plot ────────────────────────────────────────────
    ax2.plot(history.history["loss"],     label="Train Loss", color="#3498db")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#e67e22")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_name = f"training_history_{phase_name.replace(' ', '_').lower()}.png"
    plt.savefig(save_name, dpi=100)
    plt.show()
    print(f"✅ Chart saved as '{save_name}'")

    return save_name


def train():
    """
    Main training pipeline.
    
    Phase 1: Train only the top (classification) layers.
             Base MobileNetV2 is frozen.
    Phase 2: Fine-tune — unfreeze top layers of base model.
             Use a very low learning rate.
    """
    print("=" * 55)
    print("  PHASE 4: Model Training")
    print("=" * 55)

    # ── Step 1: Load data ────────────────────────────────────
    print("\n📂 Loading dataset...")
    train_gen, val_gen, class_indices = create_data_generators()

    # ── Step 2: Build model ──────────────────────────────────
    if USE_TRANSFER:
        print("\n🧠 Building MobileNetV2 Transfer Learning model...")
        model, base_model = build_model_transfer()
    else:
        print("\n🧠 Building CNN from scratch...")
        model = build_model_scratch()
        base_model = None

    model = compile_model(model, learning_rate=0.001)
    print_model_summary(model, "Waste Classifier")

    # ── Step 3: Phase 1 Training (frozen base) ───────────────
    print(f"\n🏋️  PHASE 1 TRAINING: Top layers only ({EPOCHS_PHASE1} epochs max)")
    print("   (Base MobileNetV2 is frozen — fast training!)\n")

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )

    plot_history(history1, "Phase 1 - Top Layers")

    # ── Step 4: Phase 2 Fine-tuning (only if using transfer) ─
    if USE_TRANSFER and base_model is not None:
        print(f"\n🔬 PHASE 2 FINE-TUNING: Unfreezing top 30 layers ({EPOCHS_PHASE2} epochs max)")
        print("   (Very low learning rate to avoid destroying learned features)\n")

        # Unfreeze only the LAST 30 layers of base model
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Recompile with a much lower LR for fine-tuning
        model = compile_model(model, learning_rate=1e-5)

        history2 = model.fit(
            train_gen,
            epochs=EPOCHS_PHASE2,
            validation_data=val_gen,
            callbacks=get_callbacks(),
            verbose=1
        )

        plot_history(history2, "Phase 2 - Fine Tuning")

    # ── Step 5: Final evaluation ─────────────────────────────
    print("\n📊 Final Evaluation on Validation Set:")
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    print(f"\n  ✅ Final Validation Accuracy : {accuracy * 100:.2f}%")
    print(f"  ✅ Final Validation Loss     : {loss:.4f}")

    # ── Step 6: Confirm save ──────────────────────────────────
    if os.path.exists(MODEL_SAVE_PATH):
        size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
        print(f"\n💾 Best model saved as '{MODEL_SAVE_PATH}' ({size_mb:.1f} MB)")
    else:
        # Manually save if checkpoint didn't fire
        model.save(MODEL_SAVE_PATH)
        print(f"\n💾 Model saved as '{MODEL_SAVE_PATH}'")

    print("\n🎉 Training complete! Proceed to Phase 5 (Evaluation).\n")

    return model, class_indices


# ── Run standalone ───────────────────────────────────────────
if __name__ == "__main__":
    model, class_indices = train()
