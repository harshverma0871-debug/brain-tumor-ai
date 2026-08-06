"""
src/config.py

Single source of truth for paths, image settings, and hyperparameters
used across the data pipeline, model, training, evaluation, and
Grad-CAM modules. Centralizing these avoids config drift between
scripts (e.g. app.py resizing to a different size than training did).
"""

import os

# ------------------------------------------------------------------
# Paths (all relative to the project root, so the project is portable
# across machines / Windows / Linux without editing paths by hand)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_brain_tumor_model.keras")
FINAL_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "final_brain_tumor_model.keras")

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

# Fixed, alphabetical class order. This order is what the model's
# softmax output indices correspond to (index 0 = glioma, etc.) and
# MUST stay identical between data_loader.py, train.py, evaluate.py,
# gradcam.py, and app.py.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
HEAD_EPOCHS = 15          # stage 1: train the classification head only
FINE_TUNE_EPOCHS = 10     # stage 2: unfreeze top backbone layers
FINE_TUNE_AT_LAYERS = 20  # number of top backbone layers to unfreeze
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

DROPOUT_RATE = 0.3
DENSE_UNITS = 128

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ------------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------------
BACKBONE_LAYER_NAME = "efficientnetb0"
GRADCAM_THRESHOLD = 0.25   # activations below this are suppressed (no tint)
GRADCAM_ALPHA_MAX = 0.6    # max blend strength for the hottest pixels


def ensure_dirs():
    """Create output/model directories if they don't exist yet."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
