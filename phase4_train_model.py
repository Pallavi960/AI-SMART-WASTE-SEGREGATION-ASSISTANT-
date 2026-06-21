# ============================================================
# PHASE 4 (v2): TRAIN THE MODEL — IMPROVED
# AI Smart Waste Segregation Assistant (SDG 12)
# Fixes overfitting + boosts accuracy via:
#   - Class weights (handles imbalanced trash=137 vs paper=594)
#   - Label smoothing (already in phase3 compile)
#   - Cosine decay learning rate (smoother than step drops)
#   - More conservative fine-tuning (fewer layers, lower LR)
#   - Early stopping on val_loss (catches overfitting earlier than val_accuracy)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from phase2_preprocessing import create_data_generators
from phase3_build_model import (
    build_model_transfer,
    compile_model,
    print_model_summary
)

# ─── Configuration ──────────────────────────────────────────
EPOCHS_PHASE1   = 25     # frozen base — train classification head
EPOCHS_PHASE2   = 15     # fine-tune — unfreeze top of backbone
MODEL_SAVE_PATH = "waste_model.h5"
BACKBONE        = "mobilenet"   # "mobilenet" or "efficientnet"
# ────────────────────────────────────────────────────────────


def get_callbacks(monitor="val_loss", patience=7):
    """
    Callbacks tuned to prevent overfitting:
    - Monitors val_loss (not val_accuracy) for early stopping —
      val_loss rising while val_accuracy plateaus is the earliest
      overfitting signal.
    - ModelCheckpoint saves only the genuinely best epoch.
    - ReduceLROnPlateau gives the model a chance to escape
      plateaus before giving up entirely.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def plot_history(history, phase_name="Phase 1"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Results — {phase_name}", fontsize=14)

    ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="#2ecc71")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#e74c3c")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

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


def check_overfitting(history):
    """Print a simple overfitting diagnosis based on train vs val gap."""
    train_acc = history.history["accuracy"][-1]
    val_acc   = history.history["val_accuracy"][-1]
    gap = train_acc - val_acc

    print(f"\n  📐 Train accuracy: {train_acc*100:.1f}%  |  Val accuracy: {val_acc*100:.1f}%  |  Gap: {gap*100:.1f}%")

    if gap > 0.15:
        print("  ⚠️  WARNING: Large train-val gap → model is overfitting.")
        print("     → Consider: more augmentation, smaller model, or more dropout.")
    elif gap > 0.08:
        print("  ⚠️  Mild overfitting detected — acceptable but watch closely.")
    else:
        print("  ✅ Good generalization — train/val accuracy are close.")


def train():
    print("=" * 55)
    print("  PHASE 4 (v2): Model Training — Improved")
    print("=" * 55)

    # ── Step 1: Load data ────────────────────────────────────
    print("\n📂 Loading dataset...")
    train_gen, val_gen, class_indices, class_weights = create_data_generators()

    steps_per_epoch  = train_gen.samples // train_gen.batch_size
    val_steps        = val_gen.samples // val_gen.batch_size

    # ── Step 2: Build model ──────────────────────────────────
    print(f"\n🧠 Building {BACKBONE} model with regularization...")
    model, base_model = build_model_transfer(backbone=BACKBONE)
    model = compile_model(model, learning_rate=0.001, label_smoothing=0.1)
    print_model_summary(model, "Waste Classifier v2")

    # ── Step 3: Phase 1 — train head only (base frozen) ──────
    print(f"\n🏋️  PHASE 1: Training classification head ({EPOCHS_PHASE1} epochs max)")
    print("   Base model frozen | Class weights applied | Label smoothing active\n")

    history1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        validation_steps=val_steps,
        class_weight=class_weights,     # KEY FIX: handles class imbalance
        callbacks=get_callbacks(monitor="val_loss", patience=7),
        verbose=1
    )

    plot_history(history1, "Phase 1 - Head Training")
    check_overfitting(history1)

    # ── Step 4: Phase 2 — careful fine-tuning ─────────────────
    print(f"\n🔬 PHASE 2: Fine-tuning ({EPOCHS_PHASE2} epochs max)")
    print("   Unfreezing only the LAST 20 layers (was 30 — less aggressive)")
    print("   Learning rate: 5e-6 (was 1e-5 — gentler updates)\n")

    base_model.trainable = True
    # Unfreeze fewer layers than before — this was a major overfitting
    # source previously (30 layers unfrozen on a tiny dataset).
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Keep BatchNorm layers frozen even when "unfrozen" — this is a
    # well-known fine-tuning best practice that stabilizes training.
    for layer in base_model.layers[-20:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model = compile_model(model, learning_rate=5e-6, label_smoothing=0.1)

    history2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        validation_steps=val_steps,
        class_weight=class_weights,
        callbacks=get_callbacks(monitor="val_loss", patience=5),
        verbose=1
    )

    plot_history(history2, "Phase 2 - Fine Tuning")
    check_overfitting(history2)

    # ── Step 5: Final evaluation ──────────────────────────────
    print("\n📊 Final Evaluation on Validation Set:")
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    print(f"\n  ✅ Final Validation Accuracy : {accuracy * 100:.2f}%")
    print(f"  ✅ Final Validation Loss     : {loss:.4f}")

    # ── Step 6: Confirm save ──────────────────────────────────
    if os.path.exists(MODEL_SAVE_PATH):
        size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
        print(f"\n💾 Best model saved as '{MODEL_SAVE_PATH}' ({size_mb:.1f} MB)")
    else:
        model.save(MODEL_SAVE_PATH)
        print(f"\n💾 Model saved as '{MODEL_SAVE_PATH}'")

    print("\n🎉 Training complete! Proceed to Phase 5 (Evaluation).\n")
    return model, class_indices


if __name__ == "__main__":
    model, class_indices = train()