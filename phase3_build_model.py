# ============================================================
# PHASE 3 (v2): BUILD CNN MODEL — IMPROVED
# AI Smart Waste Segregation Assistant (SDG 12)
# Adds: L2 regularization, label smoothing, better dropout schedule
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# ─── Configuration ──────────────────────────────────────────
IMG_SIZE    = (224, 224)
NUM_CLASSES = 6
L2_REG      = 1e-4    # weight decay — penalizes large weights, fights overfitting
# ────────────────────────────────────────────────────────────


def build_model_transfer(backbone="mobilenet"):
    """
    Transfer learning model with regularization to prevent overfitting.

    backbone: "mobilenet" (faster, lighter) or "efficientnet" (often more accurate)

    Key anti-overfitting techniques used:
    - L2 weight regularization on Dense layers
    - Progressive dropout (lower → higher as we go deeper)
    - BatchNormalization for stable training
    - GlobalAveragePooling instead of Flatten (far fewer params → less overfitting)
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # ── Built-in data augmentation layers (only active during training) ──
    # This adds augmentation INSIDE the model graph, so it works even
    # if you forget to set it in ImageDataGenerator, and doesn't slow
    # down inference.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # ── Backbone ──────────────────────────────────────────────
    if backbone == "efficientnet":
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
    else:
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )

    base_model.trainable = False   # frozen initially (Phase 1 training)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # ── Classification head with L2 regularization ────────────
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)        # heavy dropout right after backbone

    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)        # lighter dropout closer to output

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


def compile_model(model, learning_rate=0.001, label_smoothing=0.1):
    """
    Compile with label smoothing — this is a key overfitting fix.
    Instead of forcing the model to predict 100% confidence for the
    correct class, label smoothing softens targets (e.g. 0.9 instead
    of 1.0), which keeps the model from becoming overconfident and
    generalizes better to new images.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"]
    )
    return model


def print_model_summary(model, name="Model"):
    print("=" * 55)
    print(f"  {name} Architecture")
    print("=" * 55)
    model.summary()
    total_params = model.count_params()
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\n✅ Total parameters     : {total_params:,}")
    print(f"✅ Trainable parameters : {trainable:,}\n")


# ── Run standalone ───────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧠 Building CNN Model (MobileNetV2 + Regularization)...")
    model, base_model = build_model_transfer(backbone="mobilenet")
    model = compile_model(model)
    print_model_summary(model, "Waste Classifier v2")
    print("✅ Model built successfully! Proceed to Phase 4 (Training).\n")