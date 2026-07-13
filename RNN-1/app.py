import streamlit as st
from predict import generate_quote

st.set_page_config(
    page_title="RNN Quote Generator",
    layout="centered"
)

st.title("📝 RNN Quote Generator")

st.write(
    "Generate motivational quotes using an LSTM-based RNN."
)

prompt = st.text_input(
    "Enter starting words",
    placeholder="Believe in"
)

num_words = st.slider(
    "Number of words",
    5,
    50,
    20
)

if st.button("Generate Quote"):

    if prompt.strip() == "":
        st.warning("Enter a prompt")
    else:

        result = generate_quote(
            prompt,
            next_words=num_words
        )

        st.success("Generated Quote")
        st.write(result)