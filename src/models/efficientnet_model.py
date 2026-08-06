"""
src/models/efficientnet_model.py

Builds the EfficientNetB0-based transfer-learning classifier:

    EfficientNetB0 (frozen, ImageNet weights)
        -> GlobalAveragePooling2D
        -> Dropout(0.3)
        -> Dense(128, relu)
        -> Dropout(0.3)
        -> Dense(4, softmax)

Important implementation detail: the EfficientNetB0 backbone is kept
as a single NESTED layer (by calling `base_model(inputs)` inside the
functional API) rather than flattened into the outer model's layer
list. This matters for Grad-CAM: `model.get_layer("efficientnetb0")`
must work so gradcam.py can reach the backbone's last conv layer. If
you instead build straight off `base_model.input`/`base_model.output`,
Keras flattens the backbone's layers into the outer model and this
lookup breaks.
"""

import logging

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

from src.config import (
    IMG_SHAPE,
    DENSE_UNITS,
    DROPOUT_RATE,
    HEAD_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    BACKBONE_LAYER_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model(
    input_shape=IMG_SHAPE,
    num_classes: int = 4,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = HEAD_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build and compile the EfficientNetB0 transfer-learning model with
    the backbone frozen (stage-1 / head-only training).
    """
    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
        base_model._name = BACKBONE_LAYER_NAME  # guarantee predictable layer name
        base_model.trainable = False

        inputs = layers.Input(shape=input_shape, name="input_image")
        # training=False keeps BatchNorm layers in inference mode even if the
        # outer model.trainable flag later flips during fine-tuning setup.
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dropout(dropout_rate, name="dropout_1")(x)
        x = layers.Dense(dense_units, activation="relu", name="dense_128")(x)
        x = layers.Dropout(dropout_rate, name="dropout_2")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = models.Model(inputs, outputs, name="brain_tumor_efficientnet_b0")

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        logger.info(f"Built model with {model.count_params():,} total parameters.")
        return model

    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        raise


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    backbone_layer_name: str = BACKBONE_LAYER_NAME,
    num_layers_to_unfreeze: int = 20,
    learning_rate: float = FINE_TUNE_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Stage-2 fine-tuning: unfreeze the top `num_layers_to_unfreeze`
    layers of the EfficientNetB0 backbone and recompile with a much
    lower learning rate so we nudge the pretrained ImageNet features
    toward MRI-specific features without destroying them.

    BatchNormalization layers are deliberately kept frozen even when
    "unfrozen" numerically, since retraining BN statistics on a
    small medical dataset with a small batch size is a common source
    of fine-tuning instability.
    """
    try:
        backbone = model.get_layer(backbone_layer_name)
        backbone.trainable = True

        freeze_until = max(0, len(backbone.layers) - num_layers_to_unfreeze)
        for layer in backbone.layers[:freeze_until]:
            layer.trainable = False
        for layer in backbone.layers[freeze_until:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False  # keep BN stats stable
            else:
                layer.trainable = True

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        trainable_count = sum(
            1 for layer in backbone.layers if layer.trainable
        )
        logger.info(
            f"Fine-tuning enabled: unfroze top {num_layers_to_unfreeze} backbone "
            f"layers ({trainable_count} actually trainable after BN freeze). "
            f"Recompiled with lr={learning_rate}."
        )
        return model

    except Exception as e:
        logger.error(f"Failed to unfreeze backbone for fine-tuning: {e}")
        raise


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.models.efficientnet_model`
    m = build_model()
    m.summary()
