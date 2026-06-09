# ============================================================
# PHASE 3: BUILD CNN MODEL
# AI Smart Waste Segregation Assistant (SDG 12)
# Architecture: Transfer Learning with MobileNetV2
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ─── Configuration ──────────────────────────────────────────
IMG_SIZE   = (224, 224)
NUM_CLASSES = 6      # cardboard, glass, metal, paper, plastic, trash
# ────────────────────────────────────────────────────────────


def build_model_scratch():
    """
    Option A: Build CNN from scratch.
    Good for learning, but needs more training time.
    Expected accuracy: ~75-85%
    """
    model = models.Sequential([

        # ── Input Layer ──────────────────────────────────────
        layers.Input(shape=(224, 224, 3)),

        # ── Block 1: Detect basic features (edges, colors) ───
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2: Detect shapes and patterns ──────────────
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 3: Detect complex textures ─────────────────
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 4: High-level features ─────────────────────
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Flatten + Dense ───────────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),

        # ── Output Layer (6 waste classes) ────────────────────
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    return model


def build_model_transfer():
    """
    Option B: Transfer Learning using MobileNetV2 (RECOMMENDED).
    Pre-trained on 1.4M images → much faster + higher accuracy.
    Expected accuracy: ~88-95%
    """
    # Load MobileNetV2 WITHOUT its top classification layers
    # include_top=False → we add our own output layers
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"       # Use pre-learned weights
    )

    # Freeze base model weights (don't retrain them yet)
    base_model.trainable = False

    # Build our custom top
    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


def compile_model(model, learning_rate=0.001):
    """Compile the model with optimizer, loss, and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def print_model_summary(model, name="Model"):
    """Print model architecture summary."""
    print("=" * 55)
    print(f"  {name} Architecture")
    print("=" * 55)
    model.summary()
    total_params = model.count_params()
    print(f"\n✅ Total parameters: {total_params:,}")
    trainable = sum([
        tf.size(w).numpy() for w in model.trainable_weights
    ])
    print(f"✅ Trainable parameters: {trainable:,}\n")


# ── Run standalone ───────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧠 Building CNN Model (Transfer Learning - MobileNetV2)...")
    model, base_model = build_model_transfer()
    model = compile_model(model)
    print_model_summary(model, "MobileNetV2 Transfer Learning")
    print("✅ Model built successfully! Proceed to Phase 4 (Training).\n")
