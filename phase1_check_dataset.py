# ============================================================
# PHASE 1: CHECK DATASET STRUCTURE
# AI Smart Waste Segregation Assistant (SDG 12)
# Run this FIRST to verify your dataset is correctly placed
# ============================================================

import os

# --- Configuration ---
DATASET_DIR = "dataset"  # Folder name in your project directory

EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def check_dataset():
    print("=" * 55)
    print("  AI WASTE SEGREGATION - Dataset Check")
    print("=" * 55)

    # Check if dataset folder exists
    if not os.path.exists(DATASET_DIR):
        print(f"\n❌ ERROR: '{DATASET_DIR}' folder NOT found!")
        print("\n📥 Steps to fix:")
        print("   1. Go to Kaggle → search 'TrashNet dataset'")
        print("   2. Download and extract the ZIP file")
        print("   3. Rename the folder to 'dataset'")
        print("   4. Place it in the SAME folder as this script")
        return False

    print(f"\n✅ '{DATASET_DIR}' folder found!\n")

    total_images = 0
    all_ok = True

    for cls in EXPECTED_CLASSES:
        cls_path = os.path.join(DATASET_DIR, cls)

        if not os.path.exists(cls_path):
            print(f"  ❌ Missing class folder: {cls}/")
            all_ok = False
        else:
            # Count images
            images = [
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            count = len(images)
            total_images += count
            status = "✅" if count > 0 else "⚠️ EMPTY"
            print(f"  {status}  {cls:<12} → {count} images")

    print(f"\n  📦 Total images found: {total_images}")

    if all_ok and total_images > 0:
        print("\n✅ Dataset looks GOOD! You can proceed to Phase 2.\n")
        return True
    else:
        print("\n⚠️  Fix the issues above before training.\n")
        return False


if __name__ == "__main__":
    check_dataset()
