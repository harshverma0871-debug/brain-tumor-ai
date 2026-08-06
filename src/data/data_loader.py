"""
src/data/data_loader.py

Loads the brain MRI dataset from disk, encodes labels, performs a
stratified 70/15/15 train/val/test split, and builds tf.data.Dataset
pipelines with resizing + normalization baked in.

Supported folder layouts (auto-detected):

  1. Flat layout:
       dataset/glioma/*.jpg
       dataset/meningioma/*.jpg
       dataset/notumor/*.jpg
       dataset/pituitary/*.jpg

  2. Kaggle-style pre-split layout (e.g. "Brain Tumor MRI Dataset"):
       dataset/Training/glioma/*.jpg
       dataset/Testing/glioma/*.jpg
       ... etc for all 4 classes under both Training/ and Testing/

Either layout is scanned recursively and every image is pooled
together, then we perform our OWN stratified 70/15/15 split on the
combined pool. This keeps the split ratio and randomization
consistent and reproducible regardless of how the source dataset
happened to be organized.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    IMG_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class DatasetSplits:
    """Container for the three tf.data.Dataset pipelines plus raw
    file lists (the raw lists are handy for evaluate.py / confusion
    matrices, since we need the true labels in a plain numpy array).
    """
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    test_paths: List[str]
    test_labels: np.ndarray
    class_names: List[str]


def _find_images_by_class(root_dir: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    """
    Recursively walk root_dir. Any file that lives inside a directory
    whose name (case-insensitive) matches one of class_names is
    collected, regardless of how deeply nested it is (handles both
    the flat layout and the Training/Testing pre-split layout).
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {root_dir}\n"
            f"Expected class subfolders for: {class_names}"
        )

    class_lookup = {c.lower(): i for i, c in enumerate(class_names)}
    paths, labels = [], []

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath).lower()
        if folder_name not in class_lookup:
            continue
        label = class_lookup[folder_name]
        for fname in filenames:
            if fname.lower().endswith(VALID_EXTENSIONS):
                paths.append(os.path.join(dirpath, fname))
                labels.append(label)

    if not paths:
        raise ValueError(
            f"No images found under '{root_dir}' for classes {class_names}. "
            f"Check that folder names match exactly (case-insensitive) "
            f"and contain .jpg/.jpeg/.png files."
        )

    logger.info(f"Found {len(paths)} images across {len(class_names)} classes in '{root_dir}'.")
    for i, cls in enumerate(class_names):
        count = sum(1 for l in labels if l == i)
        logger.info(f"  {cls}: {count} images")

    return paths, labels


def _stratified_split(
    paths: List[str], labels: List[int]
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """70/15/15 stratified split (stratified = each class keeps the
    same proportion in train/val/test as it has overall)."""
    assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 1e-6, \
        "TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must sum to 1.0"

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        train_size=TRAIN_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # temp is (val + test) = VAL_SPLIT + TEST_SPLIT of the total.
    # We need val's share *within* temp.
    relative_val_size = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=relative_val_size,
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    logger.info(
        f"Split sizes -> train: {len(train_paths)}, "
        f"val: {len(val_paths)}, test: {len(test_paths)}"
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def _decode_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """Read a file, decode as RGB, resize, and apply EfficientNet's
    expected preprocessing (scales pixels to the range EfficientNet
    was pretrained on). This function runs inside the tf.data graph.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor):
    """Light, medically-plausible augmentation for the training set
    only. Avoids aggressive color jitter, since MRI intensity carries
    diagnostic meaning."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def _make_dataset(
    paths: List[str],
    labels: List[int],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    paths_t = tf.constant(paths)
    labels_t = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_datasets(
    dataset_dir: str,
    class_names: List[str] = CLASS_NAMES,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
) -> DatasetSplits:
    """
    Main entry point. Scans `dataset_dir`, builds a stratified
    70/15/15 split, and returns ready-to-use tf.data pipelines for
    train/val/test plus the raw test paths/labels (useful for
    evaluate.py's confusion matrix, which needs ground-truth labels
    in a plain array aligned with predictions).
    """
    try:
        paths, labels = _find_images_by_class(dataset_dir, class_names)
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
            _stratified_split(paths, labels)

        train_ds = _make_dataset(train_paths, train_labels, batch_size, shuffle=True, augment=augment_train)
        val_ds = _make_dataset(val_paths, val_labels, batch_size, shuffle=False, augment=False)
        test_ds = _make_dataset(test_paths, test_labels, batch_size, shuffle=False, augment=False)

        return DatasetSplits(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            test_paths=test_paths,
            test_labels=np.array(test_labels),
            class_names=class_names,
        )

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while building dataset pipelines: {e}")
        raise


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.data.data_loader`
    from src.config import DATASET_DIR

    splits = load_datasets(DATASET_DIR)
    for images, batch_labels in splits.train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", batch_labels.shape)
        print("Sample labels:", batch_labels.numpy()[:8])
