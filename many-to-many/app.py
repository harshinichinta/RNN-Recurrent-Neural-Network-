import streamlit as st
import pandas as pd

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="English to French Translator",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 English → French Translator")
st.write("Translation using the English-French dataset")

# ---------------------------------------
# Load Dataset
# ---------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("eng_-french.csv")
    return df

try:
    df = load_data()

    st.success("Dataset Loaded Successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")

    english = st.text_input("Enter English Sentence")

    if st.button("Translate"):

        result = df[
            df["English words/sentences"].str.lower()
            == english.lower()
        ]

        if len(result) > 0:
            french = result.iloc[0]["French words/sentences"]

            st.success("Translation Found")

            st.write("### English")
            st.info(english)

            st.write("### French")
            st.success(french)

        else:
            st.error("Translation not found in dataset.")

    st.markdown("---")

    if st.checkbox("Show Complete Dataset"):
        st.dataframe(df)

except FileNotFoundError:
    st.error("eng_-french.csv not found.")