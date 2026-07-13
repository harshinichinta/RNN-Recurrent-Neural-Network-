import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Page Config
st.set_page_config(page_title="MNIST RNN Classifier", layout="wide")

st.title("MNIST Handwritten Digit Classification using RNN")


# Load Model
@st.cache_resource
def load_rnn_model():
    return load_model("model/rnn_mnist_model.keras")


model = load_rnn_model()

# Load Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Dataset Info
st.header("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("Training Data Shape:")
    st.code(str(X_train.shape))

with col2:
    st.write("Testing Data Shape:")
    st.code(str(X_test.shape))

st.write("Training Labels Shape:")
st.code(str(y_train.shape))

# Display Sample Images
st.header("Sample Images")

fig, axes = plt.subplots(1, 5, figsize=(10, 3))

for i in range(5):
    axes[i].imshow(X_train[i], cmap="gray")
    axes[i].set_title(f"Label: {y_train[i]}")
    axes[i].axis("off")

st.pyplot(fig)

# Prediction Section
st.header("Test Prediction")

index = st.slider("Select Test Image", 0, len(X_test) - 1, 0)

image = X_test[index]

fig2, ax = plt.subplots()
ax.imshow(image, cmap="gray")
ax.axis("off")

st.pyplot(fig2)

prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)

predicted_digit = np.argmax(prediction)

st.subheader("Prediction")

st.success(f"Predicted Digit: {predicted_digit}")

st.info(f"Actual Digit: {y_test[index]}")

if predicted_digit == y_test[index]:
    st.success("Prediction Correct")
else:
    st.error("Prediction Incorrect")