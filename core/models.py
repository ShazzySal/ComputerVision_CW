"""Model construction and checkpoint loading for RetinaGuard AI."""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3

from core.config import AppConfig, PROJECT_ROOT


def build_classifier():
    """Construct and load the EfficientNetB3 DR classifier."""
    base_m = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3),
    )
    inputs = keras.Input(shape=(AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3))
    x = base_m(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(AppConfig.NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RetinaTrace_EfficientNetB3")

    if not os.path.exists(AppConfig.WEIGHTS_PATH):
        raise FileNotFoundError(f"Required classifier checkpoint not found: {AppConfig.WEIGHTS_PATH}")
    try:
        model.load_weights(AppConfig.WEIGHTS_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load required classifier checkpoint '{AppConfig.WEIGHTS_PATH}'.") from exc
    print(f"[Model] Fine-tuned checkpoint loaded successfully from {AppConfig.WEIGHTS_PATH}.")
    return model, base_m


def build_gradcam_model(full_model, base_model):
    """Build a model exposing final convolutional features and predictions."""
    conv_layer = base_model.get_layer("top_activation")
    base_sub = keras.Model(inputs=base_model.inputs, outputs=[conv_layer.output, base_model.output])
    cam_in = keras.Input(shape=(AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3))
    conv_output, base_output = base_sub(cam_in)
    x = full_model.get_layer("gap")(base_output)
    x = full_model.get_layer("head_bn")(x)
    x = full_model.get_layer("head_dense")(x)
    x = full_model.get_layer("head_dropout")(x)
    predictions = full_model.get_layer("predictions")(x)
    return keras.Model(inputs=cam_in, outputs=[conv_output, predictions])


def build_auxiliary_unet():
    """Construct and load the auxiliary U-Net lesion segmentation model."""
    inputs = keras.Input(shape=(AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3))
    c1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    bridge = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(p2)
    u2 = layers.UpSampling2D((2, 2))(bridge)
    cat2 = layers.concatenate([u2, c2])
    d2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(cat2)
    u1 = layers.UpSampling2D((2, 2))(d2)
    cat1 = layers.concatenate([u1, c1])
    d1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(cat1)
    output = layers.Conv2D(1, (1, 1), activation="sigmoid")(d1)
    model = keras.Model(inputs=inputs, outputs=output, name="Auxiliary_UNet")

    if not os.path.exists(AppConfig.UNET_WEIGHTS_PATH):
        raise FileNotFoundError(f"Required U-Net checkpoint not found: {AppConfig.UNET_WEIGHTS_PATH}")
    try:
        model.load_weights(AppConfig.UNET_WEIGHTS_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load required U-Net checkpoint '{AppConfig.UNET_WEIGHTS_PATH}'.") from exc
    print(f"[U-Net] Auxiliary lesion segmentation weights loaded successfully from {AppConfig.UNET_WEIGHTS_PATH}.")
    return model


def build_deep_clinical_unet(input_shape: tuple = (AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3)) -> keras.Model:
    """Construct a full 4-level deep U-Net architecture with multi-scale skip connections.
    
    Architectural Specification:
    - 4-Stage Hierarchical Encoder: (32 -> 64 -> 128 -> 256 filters)
    - Double-convolution blocks with Batch Normalization and ReLU activations
    - Latent Bottleneck: 512 filters with Spatial Dropout (rate=0.40)
    - 4-Stage Transposed Convolution Decoder with matching skip connections
    - Sigmoid 1x1 output layer for pixel-level binary lesion segmentation
    """
    def double_conv_block(x, filters, name_prefix):
        x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name_prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", name=f"{name_prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
        return x

    inputs = keras.Input(shape=input_shape, name="fundus_image_input")

    # ── 4-Stage Encoder ─────────────────────────────────────────────
    # Stage 1: 224x224
    c1 = double_conv_block(inputs, 32, "enc1")
    p1 = layers.MaxPooling2D((2, 2), name="enc1_pool")(c1)  # 112x112

    # Stage 2: 112x112
    c2 = double_conv_block(p1, 64, "enc2")
    p2 = layers.MaxPooling2D((2, 2), name="enc2_pool")(c2)  # 56x56

    # Stage 3: 56x56
    c3 = double_conv_block(p2, 128, "enc3")
    p3 = layers.MaxPooling2D((2, 2), name="enc3_pool")(c3)  # 28x28

    # Stage 4: 28x28
    c4 = double_conv_block(p3, 256, "enc4")
    p4 = layers.MaxPooling2D((2, 2), name="enc4_pool")(c4)  # 14x14

    # ── Latent Bottleneck (14x14) ───────────────────────────────────
    b = double_conv_block(p4, 512, "bottleneck")
    b = layers.Dropout(0.40, name="bottleneck_dropout")(b)

    # ── 4-Stage Decoder with Skip Connections ───────────────────────
    # Stage 4 Up: 14x14 -> 28x28
    u4 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same", name="dec4_up")(b)
    cat4 = layers.concatenate([u4, c4], name="dec4_concat")
    d4 = double_conv_block(cat4, 256, "dec4")

    # Stage 3 Up: 28x28 -> 56x56
    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", name="dec3_up")(d4)
    cat3 = layers.concatenate([u3, c3], name="dec3_concat")
    d3 = double_conv_block(cat3, 128, "dec3")

    # Stage 2 Up: 56x56 -> 112x112
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", name="dec2_up")(d3)
    cat2 = layers.concatenate([u2, c2], name="dec2_concat")
    d2 = double_conv_block(cat2, 64, "dec2")

    # Stage 1 Up: 112x112 -> 224x224
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same", name="dec1_up")(d2)
    cat1 = layers.concatenate([u1, c1], name="dec1_concat")
    d1 = double_conv_block(cat1, 32, "dec1")

    # Final pixel-wise probability mask
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="lesion_segmentation_output")(d1)

    return keras.Model(inputs=inputs, outputs=outputs, name="RetinaTrace_Deep_Clinical_UNet")


def load_reference_embeddings():
    """Load a structurally valid embedding library, even if gallery images are unavailable."""
    if os.path.exists(AppConfig.EMBEDDINGS_PATH):
        try:
            data = np.load(AppConfig.EMBEDDINGS_PATH, allow_pickle=True)
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            labels = np.asarray(data["labels"])
            filepaths = []
            for path in data["filepaths"]:
                path = str(path)
                candidate = path if os.path.isabs(path) else os.path.join(str(PROJECT_ROOT), path)
                filepaths.append(candidate)
            if embeddings.ndim != 2 or embeddings.shape[1] != 256:
                raise ValueError("reference embeddings must have shape (n, 256)")
            if len(embeddings) != len(labels) or len(labels) != len(filepaths):
                raise ValueError("reference arrays must have equal lengths")
            if not len(embeddings):
                raise ValueError("reference embedding library is empty")
            missing = sum(not os.path.exists(path) for path in filepaths)
            if missing:
                print(f"[CBR WARNING] {missing}/{len(filepaths)} reference images are unavailable; similarity metadata remains usable.")
            return embeddings, labels, filepaths
        except Exception as exc:
            print(f"[CBR WARNING] Reference embeddings are unavailable or invalid; similar-case retrieval is disabled ({exc}).")
    return np.empty((0, 256), dtype=np.float32), np.empty((0,), dtype=np.int64), []


full_model, base_model = build_classifier()
gradcam_model = build_gradcam_model(full_model, base_model)
embedding_extractor = keras.Model(inputs=full_model.input, outputs=full_model.get_layer("head_dense").output)
unet_model = build_auxiliary_unet()
ref_embeddings, ref_labels, ref_fps = load_reference_embeddings()
