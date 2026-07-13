import streamlit as st
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="English → French Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 English → French Translator")

# -------------------------------
# LOAD DATASET
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("eng_-french.csv")

    # Keep first two columns
    df = df.iloc[:, :2]

    # Rename columns
    df.columns = ["English", "French"]

    # Clean text
    df["English"] = (
        df["English"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["French"] = (
        df["French"]
        .astype(str)
        .str.strip()
    )

    return df


df = load_data()

st.success(f"✅ Loaded {len(df):,} sentence pairs")

# -------------------------------
# USER INPUT
# -------------------------------

user_text = st.text_input(
    "Enter English Sentence",
    placeholder="Example: hi"
)

# -------------------------------
# TRANSLATE
# -------------------------------

if st.button("Translate"):

    query = user_text.lower().strip()

    if query == "":
        st.warning("Please enter a sentence.")

    else:

        # 1. Exact match
        exact_match = df[df["English"] == query]

        if not exact_match.empty:

            translation = exact_match.iloc[0]["French"]

            st.subheader("French Translation")
            st.success(translation)

        else:

            # 2. Partial match
            partial_match = df[
                df["English"].str.contains(
                    query,
                    case=False,
                    na=False
                )
            ]

            if not partial_match.empty:

                translation = partial_match.iloc[0]["French"]

                st.subheader("French Translation")
                st.success(translation)

            else:

                st.error(
                    "❌ Translation not found in dataset."
                )

# -------------------------------
# SAMPLE DATA
# -------------------------------

with st.expander("View Sample Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )