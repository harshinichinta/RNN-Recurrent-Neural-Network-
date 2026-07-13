import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "models/one_to_one_rnn_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

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

    # Resize image
    image = image.resize((28, 28))

    img = np.array(image).astype("float32") / 255.0

    # Invert if white background
    if img.mean() > 0.5:
        img = 1 - img

    img = img.reshape(1, 28, 28)

    if st.button("Predict"):

        prediction = model.predict(img)

        predicted_digit = np.argmax(prediction)

        confidence = np.max(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        st.progress(float(confidence))
