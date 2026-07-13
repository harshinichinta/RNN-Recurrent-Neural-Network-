import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Harry Potter Text Generator",
    page_icon="⚡"
)

@st.cache_resource
def load_resources():
    model = load_model("models/lstm_text_generator.keras")

    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("models/max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

st.title("⚡ Harry Potter Text Generator")

seed_text = st.text_input(
    "Enter Starting Text",
    "harry"
)

num_words = st.slider(
    "Words to Generate",
    5,
    50,
    20
)

if st.button("Generate"):

    generated = seed_text

    for _ in range(num_words):

        token_list = tokenizer.texts_to_sequences([generated])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_len - 1,
            padding="pre"
        )

        predicted = np.argmax(
            model.predict(token_list, verbose=0),
            axis=-1
        )[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        generated += " " + output_word

    st.subheader("Generated Text")
    st.write(generated)