"""
app.py

Streamlit web app for Brain Tumor Detection with Grad-CAM
explainability.

Run with:
    streamlit run app.py
"""

import importlib
import logging

import numpy as np
try:
    import streamlit as st
except ImportError as exc:
    st = None
    streamlit_import_error = exc
from PIL import Image

try:
    tf = importlib.import_module("tensorflow")
except ImportError as exc:
    tf = None
    tensorflow_import_error = exc

from src.config import BEST_MODEL_PATH, CLASS_NAMES, IMG_SIZE
from src.explainability.gradcam import generate_gradcam

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# PAGE CONFIG + DARK THEME
# ==========================================

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a modern dark theme. Streamlit's built-in theme can
# be set via .streamlit/config.toml, but we also inject CSS here so
# the app looks right even if a user runs it without that config file.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .panel-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e5e7eb;
        border-bottom: 2px solid #374151;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1a1d27;
        border: 1px solid #2d3140;
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧠 Brain Tumor Detection AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an MRI scan and receive AI-powered tumor detection '
    'with explainable visual evidence.</div>',
    unsafe_allow_html=True,
)


# ==========================================
# LOAD MODEL (cached across reruns/uploads)
# ==========================================

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        return tf.keras.models.load_model(BEST_MODEL_PATH)
    except Exception as exc:
        logger.error(f"Failed to load model from {BEST_MODEL_PATH}: {exc}")
        raise


try:
    model = load_model()
except Exception:
    st.error(
        f"Could not load the model from `{BEST_MODEL_PATH}`. "
        f"Make sure you've run training (`python -m src.training.train`) "
        f"so that file exists."
    )
    st.stop()


# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read this image file: {exc}")
        st.stop()

    # --------------------------------------
    # Preprocess (shared for prediction + Grad-CAM)
    # --------------------------------------
    try:
        img_resized = image.resize(IMG_SIZE)
        display_img = np.array(img_resized).astype("uint8")

        img_array = np.array(img_resized).astype("float32")
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        input_tensor = np.expand_dims(img_array, axis=0)
    except Exception as exc:
        st.error(f"Error preprocessing image: {exc}")
        st.stop()

    # --------------------------------------
    # Prediction
    # --------------------------------------
    try:
        with st.spinner("Running prediction..."):
            preds = model.predict(input_tensor, verbose=0)[0]
            pred_class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # --------------------------------------
    # PREDICTION PANEL
    # --------------------------------------
    st.markdown('<div class="panel-header">📊 Prediction Panel</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    with col2:
        predicted_label = CLASS_NAMES[pred_class_idx]
        display_label = "No Tumor" if predicted_label == "notumor" else predicted_label.capitalize()

        st.metric(label="Predicted Class", value=display_label)
        st.metric(label="Confidence", value=f"{confidence:.2%}")

        st.write("**Class Probabilities**")
        for i, cls in enumerate(CLASS_NAMES):
            cls_display = "No Tumor" if cls == "notumor" else cls.capitalize()
            st.write(f"{cls_display}: {preds[i]:.2%}")
            st.progress(float(preds[i]))

    # --------------------------------------
    # EXPLAINABILITY PANEL
    # --------------------------------------
    st.markdown('<div class="panel-header">🔍 Explainability Panel (Grad-CAM)</div>', unsafe_allow_html=True)

    try:
        with st.spinner("Generating Grad-CAM explanation..."):
            result = generate_gradcam(model, input_tensor, display_img, class_index=pred_class_idx)

        col3, col4, col5 = st.columns(3)
        with col3:
            st.image(result["original_image"], caption="Original MRI", use_container_width=True)
        with col4:
            st.image(result["heatmap_image"], caption="Grad-CAM Heatmap", use_container_width=True)
        with col5:
            st.image(result["overlay_image"], caption="Overlay", use_container_width=True)

        st.caption(
            "The heatmap highlights the regions of the MRI that most influenced the "
            "model's prediction. Warmer colors (red/yellow) indicate stronger influence."
        )

    except Exception as exc:
        st.warning(f"Could not generate Grad-CAM explanation: {exc}")

else:
    st.info("👆 Upload an MRI image (JPG, JPEG, or PNG) to get started.")
