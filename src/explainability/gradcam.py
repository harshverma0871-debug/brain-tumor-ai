"""
src/explainability/gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) for the
EfficientNetB0 brain tumor model.

Design notes (fixes for common Grad-CAM artifacts):
  - Targets the true last Conv2D layer INSIDE the nested
    "efficientnetb0" backbone layer (found automatically), not a
    hardcoded name that might not exist in a differently-built model.
  - Upsamples the coarse (7x7 for 224x224 input) activation map with
    cubic interpolation rather than nearest-neighbor.
  - Suppresses low-activation regions below a threshold instead of
    letting the colormap tint the entire image (this is what caused
    background/skull-exterior pixels to look artificially "hot" in
    earlier versions).
  - Blends the colormap onto the original image with per-pixel alpha
    proportional to activation strength, so only genuinely important
    regions are tinted.
"""

import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import BACKBONE_LAYER_NAME, GRADCAM_THRESHOLD, GRADCAM_ALPHA_MAX, IMG_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_last_conv_layer_name(backbone: tf.keras.Model) -> str:
    """Return the name of the last Conv2D layer inside a backbone model."""
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the backbone model.")


def _call_layer(layer: tf.keras.layers.Layer, x):
    """Call a layer in inference mode, passing training=False only if
    the layer actually accepts it (Dropout/BatchNorm do; plain Dense
    etc. also accept it harmlessly via Keras' generic Layer.__call__,
    but we guard anyway for custom/edge-case layers)."""
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def compute_heatmap(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    backbone_layer_name: str = BACKBONE_LAYER_NAME,
    class_index: int = None,
):
    """
    Run a forward + backward pass to compute the Grad-CAM heatmap for
    `class_index` (or the predicted class if None).

    Implementation note: in Keras 3, a nested submodel's internal
    layer `.output` tensors belong to that submodel's OWN standalone
    graph, not the outer model's graph -- so building a multi-output
    `tf.keras.Model(inputs=model.inputs, outputs=[backbone.get_layer(...).output, model.output])`
    raises "not connected to inputs". The robust fix is to manually
    replay the outer model's layers inside a single GradientTape,
    starting from a tensor we control (the backbone's own output,
    which -- since the backbone is built with include_top=False --
    IS already the final convolutional feature map we want to
    explain).

    Returns:
        heatmap: 2D numpy array, values in [0, 1].
        class_index: the class the heatmap explains.
        predictions: raw softmax output for the input.
    """
    try:
        backbone = model.get_layer(backbone_layer_name)
    except ValueError as e:
        raise ValueError(
            f"Could not find backbone layer '{backbone_layer_name}' in the model. "
            f"Available top-level layers: {[l.name for l in model.layers]}"
        ) from e

    input_tensor = tf.convert_to_tensor(input_tensor)

    with tf.GradientTape() as tape:
        conv_output = backbone(input_tensor, training=False)
        tape.watch(conv_output)

        x = conv_output
        after_backbone = False
        for layer in model.layers:
            if layer.name == backbone.name:
                after_backbone = True
                continue
            if not after_backbone or isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = _call_layer(layer, x)

        predictions = x

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise RuntimeError(
            "Gradient computation returned None. This usually means the "
            "conv layer output is disconnected from the model output in "
            "the graph -- check that the model's layer order after the "
            f"'{backbone_layer_name}' backbone is a simple linear stack."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)  # ReLU: keep only positive influence
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), class_index, predictions.numpy()[0]


def overlay_heatmap(
    display_img: np.ndarray,
    heatmap: np.ndarray,
    threshold: float = GRADCAM_THRESHOLD,
    alpha_max: float = GRADCAM_ALPHA_MAX,
    colormap: int = cv2.COLORMAP_JET,
):
    """
    Produce two outputs:
      - colored_heatmap: the heatmap alone, resized + colorized (for
        display as its own panel).
      - overlay: the heatmap intensity-blended on top of display_img.

    display_img must be HxWx3 uint8 RGB.
    """
    h, w = display_img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    suppressed = heatmap_resized.copy()
    suppressed[suppressed < threshold] = 0
    if suppressed.max() > 0:
        suppressed = suppressed / suppressed.max()

    heatmap_uint8 = np.uint8(255 * suppressed)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    alpha = (suppressed * alpha_max)[..., np.newaxis]
    display_f = display_img.astype("float32")
    colored_f = colored.astype("float32")
    overlay = display_f * (1 - alpha) + colored_f * alpha
    overlay = np.uint8(np.clip(overlay, 0, 255))

    return colored, overlay


def generate_gradcam(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    display_img: np.ndarray,
    class_index: int = None,
):
    """
    High-level convenience function used by both app.py and the
    standalone CLI below. Returns a dict with everything the UI /
    caller needs.
    """
    heatmap, resolved_class_index, predictions = compute_heatmap(model, input_tensor, class_index=class_index)
    colored_heatmap, overlay = overlay_heatmap(display_img, heatmap)

    return {
        "heatmap_raw": heatmap,
        "heatmap_image": colored_heatmap,
        "overlay_image": overlay,
        "original_image": display_img,
        "class_index": resolved_class_index,
        "predictions": predictions,
    }


def save_gradcam_outputs(result: dict, output_dir: str, prefix: str = "gradcam"):
    """Save original / heatmap / overlay as three separate images, as
    required by the project spec."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for key, fname in [
        ("original_image", f"{prefix}_original.png"),
        ("heatmap_image", f"{prefix}_heatmap.png"),
        ("overlay_image", f"{prefix}_overlay.png"),
    ]:
        path = os.path.join(output_dir, fname)
        Image.fromarray(result[key]).save(path)
        paths[key] = path
        logger.info(f"Saved {key} to {path}")
    return paths


def load_and_preprocess_for_gradcam(image_path: str):
    """Load an image from disk and produce both the model input
    tensor and a display-ready uint8 RGB array, using the same
    resize/preprocess settings as training and app.py."""
    image = Image.open(image_path).convert("RGB")
    img_resized = image.resize(IMG_SIZE)

    display_img = np.array(img_resized).astype("uint8")

    img_array = np.array(img_resized).astype("float32")
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    input_tensor = np.expand_dims(img_array, axis=0)

    return input_tensor, display_img


if __name__ == "__main__":
    # CLI usage: python -m src.explainability.gradcam <image_path> [output_dir]
    import sys
    from src.config import BEST_MODEL_PATH, OUTPUTS_DIR

    if len(sys.argv) < 2:
        print("Usage: python -m src.explainability.gradcam <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUTS_DIR

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    input_tensor, display_img = load_and_preprocess_for_gradcam(image_path)
    result = generate_gradcam(model, input_tensor, display_img)
    save_gradcam_outputs(result, output_dir)

    print(f"Predicted class index: {result['class_index']}")
    print(f"Predictions: {result['predictions']}")
