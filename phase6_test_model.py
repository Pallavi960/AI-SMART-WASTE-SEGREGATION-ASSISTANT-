# ============================================================
# PHASE 6 (v2): TEST MODEL ON NEW IMAGES — IMPROVED
# AI Smart Waste Segregation Assistant (SDG 12)
# Adds: Test-Time Augmentation (TTA) for more reliable predictions
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageEnhance
import tensorflow as tf
import sys
import os

# ─── Configuration ──────────────────────────────────────────
MODEL_PATH = "waste_model.h5"
IMG_SIZE   = (224, 224)
CLASSES    = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
USE_TTA    = True     # Test-Time Augmentation — averages predictions
                       # over several slightly-altered versions of the
                       # same image for a more robust, often more
                       # accurate final prediction.
# ────────────────────────────────────────────────────────────

WASTE_INFO = {
    "cardboard": {
        "emoji": "📦", "bin_color": "Blue Recycling Bin", "bin_hex": "#3498db",
        "advice": [
            "Flatten boxes before recycling",
            "Remove any plastic tape or staples",
            "Keep dry — wet cardboard cannot be recycled",
            "Break down large boxes to save space"
        ],
        "impact": "Recycling 1 tonne of cardboard saves 17 trees 🌳",
        "sdg_message": "SDG 12: Reduces deforestation and production waste"
    },
    "glass": {
        "emoji": "🍶", "bin_color": "Green Glass Bin", "bin_hex": "#27ae60",
        "advice": [
            "Rinse bottles and jars before recycling",
            "Remove metal lids and recycle separately",
            "Do NOT put broken glass in recycling bin — wrap it first",
            "Glass can be recycled infinitely without quality loss"
        ],
        "impact": "Recycling glass uses 40% less energy than making new glass ⚡",
        "sdg_message": "SDG 12: Conserves raw materials and saves energy"
    },
    "metal": {
        "emoji": "🥫", "bin_color": "Yellow Metal Bin", "bin_hex": "#f39c12",
        "advice": [
            "Rinse cans to remove food residue",
            "Crush cans to save space if possible",
            "Aluminium and steel are both recyclable",
            "Remove paper labels if possible"
        ],
        "impact": "Recycling aluminium uses 95% less energy than making it new ⚡",
        "sdg_message": "SDG 12: Reduces mining impact and industrial waste"
    },
    "paper": {
        "emoji": "📄", "bin_color": "Blue Recycling Bin", "bin_hex": "#3498db",
        "advice": [
            "Keep paper dry and clean",
            "Remove plastic windows from envelopes",
            "Shredded paper can be composted if not recycled",
            "Pizza boxes with grease → compost, not recycle"
        ],
        "impact": "Recycling 1 tonne of paper saves 380 gallons of oil 🛢️",
        "sdg_message": "SDG 12: Saves forests and reduces chemical pollution"
    },
    "plastic": {
        "emoji": "🧴", "bin_color": "Yellow Recycling Bin", "bin_hex": "#e74c3c",
        "advice": [
            "Check the recycling number (♻ 1–7) on the bottom",
            "Rinse containers before recycling",
            "Plastic bags → take to store drop-off, not curbside",
            "Avoid single-use plastics when possible (SDG 12)"
        ],
        "impact": "Recycling plastic prevents 1000s of years in landfill 🌍",
        "sdg_message": "SDG 12 + SDG 13: Reduces plastic pollution and CO₂ emissions"
    },
    "trash": {
        "emoji": "🗑️", "bin_color": "Black General Waste Bin", "bin_hex": "#7f8c8d",
        "advice": [
            "This item likely cannot be recycled",
            "Place in general waste bin",
            "Reduce use of non-recyclable items in the future",
            "Consider alternatives: reusable products where possible"
        ],
        "impact": "Reducing general waste helps decrease landfill overload 🏭",
        "sdg_message": "SDG 12: Encourages mindful consumption habits"
    }
}


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


def generate_tta_versions(img: Image.Image):
    """
    Create several augmented versions of the same image for
    Test-Time Augmentation. Averaging predictions across these
    reduces noise from any single odd crop/lighting/angle and
    typically improves real-world accuracy by 1-3%.
    """
    versions = []

    # Original
    versions.append(img)
    # Horizontal flip
    versions.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    # Slightly brighter
    versions.append(ImageEnhance.Brightness(img).enhance(1.15))
    # Slightly darker
    versions.append(ImageEnhance.Brightness(img).enhance(0.85))
    # Slightly zoomed (center crop 90%)
    w, h = img.size
    crop = img.crop((int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95)))
    versions.append(crop)

    return versions


def predict_waste(image_path, model, use_tta=USE_TTA):
    """
    Run prediction on a single image.
    If use_tta=True, averages predictions over multiple augmented
    versions for a more robust result.
    """
    original_img = Image.open(image_path).convert("RGB")

    if use_tta:
        versions = generate_tta_versions(original_img)
        all_preds = []
        for v in versions:
            v_resized = v.resize(IMG_SIZE)
            v_array = np.array(v_resized) / 255.0
            v_array = np.expand_dims(v_array, axis=0)
            pred = model.predict(v_array, verbose=0)[0]
            all_preds.append(pred)
        predictions = np.mean(all_preds, axis=0)   # average over all versions
    else:
        img_array, _ = preprocess_image(image_path)
        predictions = model.predict(img_array, verbose=0)[0]

    predicted_idx   = np.argmax(predictions)
    predicted_class = CLASSES[predicted_idx]
    confidence      = predictions[predicted_idx] * 100

    return predicted_class, confidence, predictions, original_img


def display_result(image_path, model):
    predicted_class, confidence, all_probs, original_img = predict_waste(image_path, model)

    info  = WASTE_INFO[predicted_class]
    top_3 = np.argsort(all_probs)[::-1][:3]

    print("\n" + "=" * 55)
    print("  🌍 AI SMART WASTE SEGREGATION ASSISTANT")
    print(f"  Powered by CNN | TTA: {'ON' if USE_TTA else 'OFF'} | SDG 12")
    print("=" * 55)
    print(f"\n  {info['emoji']}  Waste Type  : {predicted_class.upper()}")
    print(f"  🎯  Confidence : {confidence:.1f}%")
    print(f"  🗑️   Dispose In : {info['bin_color']}")
    print(f"\n  📋  Disposal Advice:")
    for tip in info["advice"]:
        print(f"      • {tip}")
    print(f"\n  🌱  Impact     : {info['impact']}")
    print(f"  🌐  SDG Note   : {info['sdg_message']}")
    print("\n  📊  Top Predictions:")
    for i in top_3:
        bar_len = int(all_probs[i] * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        marker  = " ← PREDICTED" if i == np.argmax(all_probs) else ""
        print(f"      {CLASSES[i]:<12} [{bar}] {all_probs[i]*100:5.1f}%{marker}")
    print()

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_img)
    ax1.set_title("Input Image", color="white", fontsize=13)
    ax1.axis("off")
    color_patch = mpatches.Patch(color=info["bin_hex"], label=info["bin_color"])
    ax1.legend(handles=[color_patch], loc="lower right", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#16213e")
    colors = [info["bin_hex"] if c == predicted_class else "#4a4a6a" for c in CLASSES]
    bars = ax2.barh(CLASSES, all_probs * 100, color=colors, edgecolor="none")
    ax2.set_title(f"Prediction: {predicted_class.upper()} ({confidence:.1f}%)", color="white", fontsize=13)
    ax2.set_xlabel("Confidence (%)", color="white")
    ax2.tick_params(colors="white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color("#555")
    ax2.spines["left"].set_color("#555")
    ax2.set_xlim(0, 105)
    for bar, prob in zip(bars, all_probs):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                  f"{prob*100:.1f}%", va="center", color="white", fontsize=10)

    plt.suptitle("🌍 AI Smart Waste Segregation Assistant — SDG 12", color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("✅ Result saved as 'prediction_result.png'")

    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python phase6_test_model.py <path_to_image>\n")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"\n❌ Image not found: '{image_path}'\n")
        sys.exit(1)

    print(f"\n📂 Loading model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded!")

    display_result(image_path, model)