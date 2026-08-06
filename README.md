# 🧠 Brain Tumor Detection AI with Explainable AI (Grad-CAM)

An end-to-end deep learning system that classifies brain MRI scans into
four categories — **glioma**, **meningioma**, **pituitary tumor**, or
**no tumor** — using a fine-tuned EfficientNetB0, and explains every
prediction visually with **Grad-CAM**. Includes a Streamlit web app for
interactive use.

---

## 1. Project Overview

| | |
|---|---|
| **Task** | 4-class MRI image classification |
| **Model** | EfficientNetB0 (ImageNet transfer learning) + custom classification head |
| **Explainability** | Grad-CAM (gradient-weighted class activation mapping) |
| **Interface** | Streamlit web app |
| **Framework** | TensorFlow 2.x / Keras 3 |

Pipeline: `dataset -> data_loader -> EfficientNetB0 model -> train -> evaluate -> Grad-CAM -> Streamlit app`.

---

## 2. Dataset Format

Place your dataset under `dataset/`. **Two layouts are supported automatically** —
`data_loader.py` scans recursively and pools every image it finds under a folder
named after one of the four classes, so either layout works with zero config changes:

**Flat layout:**
```
dataset/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

**Pre-split layout** (e.g. the common Kaggle "Brain Tumor MRI Dataset", which
ships as `Training/` + `Testing/` folders):
```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Whichever layout you use, `data_loader.py` pools **all** images together and
performs its own **stratified 70% / 15% / 15% train / validation / test split**
(stratified = each class keeps the same proportion in every split), so the
split is consistent and reproducible (fixed random seed) regardless of how the
source data was originally organized.

Supported image formats: `.jpg`, `.jpeg`, `.png`.

---

## 3. Project Structure

```
brain-tumor-ai/
│
├── app.py                          # Streamlit web app
├── dataset/                        # Place your MRI dataset here (see above)
├── saved_models/                   # Trained models land here
│   ├── best_brain_tumor_model.keras    # Best checkpoint (lowest val_loss)
│   └── final_brain_tumor_model.keras   # Model state at end of training
├── outputs/                        # Plots, confusion matrix, reports, Grad-CAM samples
├── requirements.txt
├── README.md
│
└── src/
    ├── config.py                   # Central config: paths, hyperparameters, class names
    ├── data/
    │   └── data_loader.py          # Loading, stratified split, tf.data pipeline
    ├── models/
    │   └── efficientnet_model.py   # EfficientNetB0 architecture + fine-tuning helper
    ├── explainability/
    │   └── gradcam.py              # Grad-CAM heatmap + overlay generation
    └── training/
        ├── train.py                # Two-stage training (head, then fine-tune)
        └── evaluate.py             # Test-set metrics + confusion matrix
```

---

## 4. Installation

Requires Python 3.10–3.12.

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **GPU (optional but recommended for training):** the pinned `tensorflow`
> package includes GPU support on Linux/Windows if you have a compatible
> NVIDIA driver + CUDA/cuDNN installed. CPU-only training works too, just
> slower. Run `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
> to check what TensorFlow can see.

---

## 5. Training

1. Put your dataset in `dataset/` (see format above).
2. Run:

```bash
python -m src.training.train
```

Optional flags:

```bash
python -m src.training.train \
    --dataset_dir dataset \
    --batch_size 32 \
    --head_epochs 15 \
    --fine_tune_epochs 10 \
    --fine_tune_layers 20

# To skip stage-2 fine-tuning entirely (head-only training):
python -m src.training.train --no_fine_tune
```

**What happens:**
- **Stage 1 (head-only):** EfficientNetB0 backbone frozen; only the
  `GlobalAveragePooling2D -> Dropout -> Dense(128) -> Dropout -> Dense(4)`
  head is trained. Fast, stabilizes the new head before touching pretrained
  weights.
- **Stage 2 (fine-tuning):** the top N layers of the backbone are unfrozen
  and trained at a much lower learning rate (`1e-5` by default) to adapt
  ImageNet features to MRI-specific features. BatchNorm layers stay frozen
  during this stage for training stability.
- **Callbacks:** `EarlyStopping` (restores best weights), `ReduceLROnPlateau`,
  and `ModelCheckpoint` (saves the best model by validation loss) are active
  throughout both stages.
- **Outputs:**
  - `saved_models/best_brain_tumor_model.keras` — best checkpoint (used by the app)
  - `saved_models/final_brain_tumor_model.keras` — model at the end of training
  - `outputs/accuracy_curve.png`, `outputs/loss_curve.png`

---

## 6. Evaluation

```bash
python -m src.training.evaluate
```

Runs the best saved model against the held-out test split and prints/saves:
- Overall accuracy
- Weighted precision, recall, F1 score
- Full per-class `classification_report`
- `outputs/confusion_matrix.png`
- `outputs/evaluation_report.txt`

---

## 7. Running the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

**Features:**
- Upload an MRI scan (`.jpg`, `.jpeg`, `.png`)
- **Prediction panel:** predicted class, confidence, and per-class probability
  bars (`st.metric` + `st.progress`)
- **Explainability panel:** original MRI, Grad-CAM heatmap, and overlay
  side by side

The app requires `saved_models/best_brain_tumor_model.keras` to exist —
run training first.

---

## 8. How Grad-CAM Works Here

Grad-CAM (`src/explainability/gradcam.py`) explains a prediction by:

1. Running a forward pass and capturing the activations of the **last
   convolutional layer inside the EfficientNetB0 backbone** (auto-detected —
   no hardcoded layer name needed).
2. Computing the gradient of the predicted class's score with respect to
   those activations, and global-average-pooling the gradients into one
   importance weight per channel.
3. Weighting the activation maps by those importance weights and summing
   them, then applying ReLU (only positive influence matters) to get a
   coarse heatmap (7×7 for a 224×224 input, since that's EfficientNetB0's
   final spatial resolution).
4. Upsampling that heatmap to the original image size with **cubic
   interpolation**, then **suppressing low-activation pixels below a
   threshold** and blending the colormap with **alpha proportional to
   activation strength** — this keeps irrelevant regions (like the black
   background outside the skull) close to the original image instead of
   being artificially tinted, which is a common Grad-CAM visualization bug.

You can also run Grad-CAM standalone on any image:

```bash
python -m src.explainability.gradcam path/to/mri_image.jpg outputs/
```

This saves three separate files: `gradcam_original.png`, `gradcam_heatmap.png`,
and `gradcam_overlay.png`.

---

## 9. Example Outputs

After training and evaluation, you should find in `outputs/`:

| File | Description |
|---|---|
| `accuracy_curve.png` | Train vs. validation accuracy per epoch |
| `loss_curve.png` | Train vs. validation loss per epoch |
| `confusion_matrix.png` | 4×4 confusion matrix heatmap on the test set |
| `evaluation_report.txt` | Accuracy, precision, recall, F1, per-class breakdown |
| `gradcam_*.png` | Original / heatmap / overlay for any image you run Grad-CAM on |

---

## 10. Notes on Code Quality

- **Modular:** data, model, training, evaluation, and explainability are
  fully separated and independently runnable modules.
- **Config-driven:** all paths, image size, split ratios, and hyperparameters
  live in `src/config.py` — no magic numbers duplicated across files.
- **Exception handling:** every entry point (`train.py`, `evaluate.py`,
  `gradcam.py`, `app.py`, `data_loader.py`) wraps its core logic in
  try/except with logged, actionable error messages instead of raw
  tracebacks.
- **Cross-platform:** all paths are built with `os.path.join` off
  `PROJECT_ROOT`, and Matplotlib uses the `Agg` backend so plotting works
  headless on Windows/servers without a display.
- **TensorFlow 2.x / Keras 3 compatible:** uses `.keras` model format,
  `tf.keras.applications.EfficientNetB0`, and the functional API throughout.
