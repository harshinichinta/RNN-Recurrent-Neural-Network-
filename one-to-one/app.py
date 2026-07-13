import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")

# -----------------------------
# Model Path
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "one_to_one_rnn_model.keras"

st.write("Model path:", MODEL_PATH)
st.write("Model exists:", MODEL_PATH.exists())

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as e:
        st.error(f"Error loading model:\n{e}")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error("Unable to load the trained model.")
        st.exception(e)
        st.stop()

model = load_model()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=250)

    image = image.resize((28, 28))

    img = np.array(image).astype("float32") / 255.0

    # Invert if background is white
    if img.mean() > 0.5:
        img = 1 - img

    img = img.reshape(1, 28, 28)

    if st.button("Predict"):

        prediction = model.predict(img, verbose=0)

        digit = np.argmax(prediction)

        confidence = float(np.max(prediction))

        st.success(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        st.progress(confidence)