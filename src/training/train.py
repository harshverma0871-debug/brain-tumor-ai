"""
src/training/train.py

Two-stage training:
  Stage 1 (head-only): backbone frozen, train GAP/Dropout/Dense head.
  Stage 2 (fine-tune):  unfreeze the top N backbone layers at a much
                         lower learning rate.

Usage:
    python -m src.training.train
    python -m src.training.train --dataset_dir path/to/dataset --head_epochs 15 --fine_tune_epochs 10
    python -m src.training.train --no_fine_tune   # stage 1 only
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend (works on servers / Windows without a display)
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required to run the training script. "
        "Install TensorFlow in your environment and try again."
    ) from exc

from src.config import (
    DATASET_DIR,
    OUTPUTS_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    CLASS_NAMES,
    BATCH_SIZE,
    HEAD_EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_AT_LAYERS,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
    ensure_dirs,
)
from src.data.data_loader import load_datasets
from src.models.efficientnet_model import build_model, unfreeze_for_fine_tuning

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_callbacks(checkpoint_path: str) -> list:
    """EarlyStopping + ReduceLROnPlateau + ModelCheckpoint, all
    watching validation loss."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def merge_histories(hist_list: list) -> dict:
    """Concatenate Keras History.history dicts from stage 1 and stage
    2 so accuracy/loss curves show one continuous timeline."""
    merged = {}
    for hist in hist_list:
        for key, values in hist.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def plot_curves(history: dict, output_dir: str, fine_tune_start_epoch: int = None):
    """Save accuracy and loss curves as separate PNGs in outputs/."""
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    if fine_tune_start_epoch is not None:
        plt.axvline(x=fine_tune_start_epoch, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    logger.info(f"Saved accuracy curve to {acc_path}")

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    if fine_tune_start_epoch is not None:
        plt.axvline(x=fine_tune_start_epoch, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    logger.info(f"Saved loss curve to {loss_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the brain tumor EfficientNetB0 classifier.")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--head_epochs", type=int, default=HEAD_EPOCHS)
    parser.add_argument("--fine_tune_epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--fine_tune_layers", type=int, default=FINE_TUNE_AT_LAYERS)
    parser.add_argument("--no_fine_tune", action="store_true", help="Skip stage 2 fine-tuning.")
    args = parser.parse_args()

    ensure_dirs()

    try:
        logger.info("Loading dataset...")
        splits = load_datasets(args.dataset_dir, class_names=CLASS_NAMES, batch_size=args.batch_size)

        logger.info("Building model (stage 1: frozen backbone)...")
        model = build_model()
        model.summary(print_fn=logger.info)

        callbacks = get_callbacks(BEST_MODEL_PATH)

        logger.info(f"Stage 1: training head for up to {args.head_epochs} epochs...")
        history_stage1 = model.fit(
            splits.train_ds,
            validation_data=splits.val_ds,
            epochs=args.head_epochs,
            callbacks=callbacks,
        )

        histories = [history_stage1]
        fine_tune_start_epoch = None

        if not args.no_fine_tune:
            logger.info(f"Stage 2: unfreezing top {args.fine_tune_layers} backbone layers for fine-tuning...")
            model = unfreeze_for_fine_tuning(model, num_layers_to_unfreeze=args.fine_tune_layers)

            fine_tune_start_epoch = len(history_stage1.history["loss"])
            total_epochs = fine_tune_start_epoch + args.fine_tune_epochs

            history_stage2 = model.fit(
                splits.train_ds,
                validation_data=splits.val_ds,
                epochs=total_epochs,
                initial_epoch=fine_tune_start_epoch,
                callbacks=callbacks,
            )
            histories.append(history_stage2)

        # Save the final (post-fine-tuning) model too. The BEST model
        # (lowest val_loss across both stages) was already saved by
        # ModelCheckpoint at BEST_MODEL_PATH.
        model.save(FINAL_MODEL_PATH)
        logger.info(f"Saved final model to {FINAL_MODEL_PATH}")
        logger.info(f"Best model (by val_loss) is at {BEST_MODEL_PATH}")

        merged_history = merge_histories(histories)
        plot_curves(merged_history, OUTPUTS_DIR, fine_tune_start_epoch)

        logger.info("Training complete.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
