# ============================================================
# PHASE 5: MODEL EVALUATION
# AI Smart Waste Segregation Assistant (SDG 12)
# Accuracy, Classification Report, Confusion Matrix
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import tensorflow as tf

from phase2_preprocessing import create_data_generators

# ─── Configuration ──────────────────────────────────────────
MODEL_PATH = "waste_model.h5"
CLASSES    = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# ────────────────────────────────────────────────────────────


def evaluate_model():
    """Full evaluation of the trained model."""
    print("=" * 55)
    print("  PHASE 5: Model Evaluation")
    print("=" * 55)

    # ── Load model ───────────────────────────────────────────
    print(f"\n📂 Loading model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded!\n")

    # ── Load validation data ─────────────────────────────────
    _, val_gen, class_indices = create_data_generators()

    # ── Predict ──────────────────────────────────────────────
    print("🔍 Running predictions on validation set...")
    val_gen.reset()

    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = val_gen.classes

    # ── Accuracy ─────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*55}\n")

    # Grade
    if accuracy >= 0.90:
        grade = "🏆 EXCELLENT — Ready for deployment!"
    elif accuracy >= 0.80:
        grade = "✅ GOOD — Acceptable for internship demo."
    elif accuracy >= 0.70:
        grade = "⚠️  FAIR — Consider more training epochs."
    else:
        grade = "❌ POOR — Check dataset and model config."

    print(f"  Grade: {grade}\n")

    # ── Per-class report ──────────────────────────────────────
    print("📋 Classification Report (per class):\n")
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(report)

    # ── Confusion Matrix ──────────────────────────────────────
    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5
    )
    plt.title("Confusion Matrix — Waste Classification", fontsize=14, pad=15)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.show()
    print("✅ Confusion matrix saved as 'confusion_matrix.png'")

    # ── Per-class accuracy bar chart ─────────────────────────
    print("\n📊 Generating per-class accuracy chart...")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    colors = ["#2ecc71" if a >= 0.80 else "#e67e22" if a >= 0.60 else "#e74c3c"
              for a in per_class_acc]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(CLASSES, per_class_acc * 100, color=colors, edgecolor="white", linewidth=1.2)
    plt.axhline(y=80, color="gray", linestyle="--", alpha=0.7, label="80% target")
    plt.title("Per-Class Accuracy", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Waste Category")
    plt.ylim([0, 105])
    plt.legend()

    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc*100:.1f}%",
            ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.savefig("per_class_accuracy.png", dpi=100)
    plt.show()
    print("✅ Per-class chart saved as 'per_class_accuracy.png'")

    print("\n🎉 Evaluation complete! Proceed to Phase 6 (Testing).\n")

    return accuracy, cm


# ── Run standalone ───────────────────────────────────────────
if __name__ == "__main__":
    evaluate_model()
