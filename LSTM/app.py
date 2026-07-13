import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Must be first Streamlit command
# -------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

MODEL_PATH = "model/imdb_lstm.keras"
METADATA_PATH = "model/imdb_metadata.pkl"

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return (
        model,
        metadata["tokenizer"],
        metadata["max_len"],
    )

model, tokenizer, max_len = load_resources()


def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    prob = float(model.predict(seq, verbose=0)[0][0])

    if prob >= 0.5:
        return "Positive 😊", prob
    else:
        return "Negative 😞", 1 - prob


st.title("🎬 IMDB Movie Review Sentiment Analysis")

review = st.text_area(
    "Enter a movie review",
    height=200,
    placeholder="Type your movie review here..."
)

if st.button("Analyze Sentiment"):
    if review.strip():
        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")
        st.success(sentiment)
        st.write(f"Confidence: **{confidence:.2%}**")
    else:
        st.warning("Please enter a review.")