import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os

# -----------------------------
# Load Model
# -----------------------------
if not os.path.exists("model.h5"):
    st.error("❌ model.h5 not found. Please run train.py first.")
    st.stop()

if not os.path.exists("tokenizer.pkl"):
    st.error("❌ tokenizer.pkl not found. Please run train.py first.")
    st.stop()

model = tf.keras.models.load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    char_to_index, index_to_char = pickle.load(f)

vocab_size = len(char_to_index)
sequence_length = 100

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Many-to-Many RNN Text Generator",
    page_icon="📖",
    layout="centered"
)

st.title("📖 Many-to-Many RNN Text Generator")
st.write("Generate Shakespeare-style text using a trained Many-to-Many RNN.")

seed = st.text_area(
    "Enter starting text",
    value="KING",
    height=150
)

num_chars = st.slider(
    "Characters to Generate",
    min_value=50,
    max_value=500,
    value=200,
    step=10
)

temperature = st.slider(
    "Temperature",
    min_value=0.2,
    max_value=1.5,
    value=1.0,
    step=0.1
)

# -----------------------------
# Sampling Function
# -----------------------------
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")

    predictions = np.log(predictions + 1e-8) / temperature

    exp_preds = np.exp(predictions)

    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, 1)

    return np.argmax(probas)

# -----------------------------
# Text Generation
# -----------------------------
def generate_text(seed_text, length):

    generated = seed_text

    for _ in range(length):

        encoded = []

        for c in generated[-sequence_length:]:

            if c in char_to_index:
                encoded.append(char_to_index[c])
            else:
                encoded.append(0)

        while len(encoded) < sequence_length:
            encoded.insert(0, 0)

        x = tf.keras.utils.to_categorical(
            encoded,
            num_classes=vocab_size
        )

        x = np.expand_dims(x, axis=0)

        prediction = model.predict(
            x,
            verbose=0
        )

        next_index = sample(
            prediction[0][-1],
            temperature
        )

        next_char = index_to_char[next_index]

        generated += next_char

    return generated

# -----------------------------
# Button
# -----------------------------
if st.button("Generate Text"):

    if len(seed.strip()) == 0:
        st.warning("Please enter some starting text.")
    else:

        with st.spinner("Generating..."):

            output = generate_text(
                seed,
                num_chars
            )

        st.success("Generation Complete!")

        st.subheader("Generated Text")

        st.write(output)