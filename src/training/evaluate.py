"""
src/training/evaluate.py

Loads the best saved model, runs it on the held-out test split, and
reports accuracy, precision, recall, F1 (per-class + weighted
average), plus a saved confusion matrix image.

Usage:
    python -m src.training.evaluate
    python -m src.training.evaluate --model_path saved_models/best_brain_tumor_model.keras
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import DATASET_DIR, OUTPUTS_DIR, BEST_MODEL_PATH, CLASS_NAMES, ensure_dirs
from src.data.data_loader import load_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset, test_labels: np.ndarray, class_names: list):
    """Run predictions on the test set and compute all metrics."""
    logger.info("Running predictions on the test set...")
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_labels

    if len(y_pred) != len(y_true):
        # Safety check: tf.data batching should preserve order since
        # test_ds was built with shuffle=False, but we verify anyway
        # rather than silently misaligning predictions with labels.
        raise ValueError(
            f"Prediction count ({len(y_pred)}) does not match label count "
            f"({len(y_true)}). Was the test dataset shuffled?"
        )

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_confusion_matrix(cm: np.ndarray, class_names: list, output_dir: str):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {path}")


def save_text_report(results: dict, output_dir: str):
    path = os.path.join(output_dir, "evaluation_report.txt")
    with open(path, "w") as f:
        f.write("BRAIN TUMOR MODEL - TEST SET EVALUATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:            {results['accuracy']:.4f}\n")
        f.write(f"Precision (weighted): {results['precision']:.4f}\n")
        f.write(f"Recall (weighted):    {results['recall']:.4f}\n")
        f.write(f"F1 Score (weighted):  {results['f1']:.4f}\n\n")
        f.write("Per-class report:\n")
        f.write(results["report"])
    logger.info(f"Saved evaluation report to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained brain tumor model on the test set.")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--model_path", type=str, default=BEST_MODEL_PATH)
    args = parser.parse_args()

    ensure_dirs()

    try:
        logger.info(f"Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)

        logger.info("Loading test dataset...")
        splits = load_datasets(args.dataset_dir, class_names=CLASS_NAMES)

        results = evaluate_model(model, splits.test_ds, splits.test_labels, splits.class_names)

        logger.info(f"Test Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"Precision (weighted): {results['precision']:.4f}")
        logger.info(f"Recall (weighted):    {results['recall']:.4f}")
        logger.info(f"F1 Score (weighted):  {results['f1']:.4f}")
        logger.info("\n" + results["report"])

        save_confusion_matrix(results["confusion_matrix"], splits.class_names, OUTPUTS_DIR)
        save_text_report(results, OUTPUTS_DIR)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
