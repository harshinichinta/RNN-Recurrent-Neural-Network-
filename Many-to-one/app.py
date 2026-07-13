# ==========================================================
# SMS Spam Detection using Simple RNN (Many-to-One)
# Part 1A
# ==========================================================

import os
import re
import pickle
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Embedding,
        SimpleRNN,
        Dense
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    TENSORFLOW_AVAILABLE = True

except Exception:

    TENSORFLOW_AVAILABLE = False

    Sequential = None
    load_model = None
    Embedding = None
    SimpleRNN = None
    Dense = None
    Tokenizer = None
    pad_sequences = None


# ==========================================================
# Configuration
# ==========================================================

MODEL = "spam_model.keras"

TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000

MAX_LEN = 50


# ==========================================================
# Clean Text
# ==========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# Train Model
# ==========================================================

def train_model():

    print("Training Dataset...\n")

    df = pd.read_csv(
        "C:\Tekworks\Recurrent-Neural-Network\Many-to-one\spam.csv",
        encoding="latin-1",
        usecols=[0, 1]
    )

    df.columns = [
        "label",
        "message"
    ]

    print(df.head())

    print("\nClass Distribution\n")

    print(df["label"].value_counts())

    # Convert Labels

    df["label"] = df["label"].map({

        "ham": 0,

        "spam": 1

    })

    # Clean Messages

    df["message"] = df["message"].apply(clean_text)

    # Tokenizer

    tokenizer = Tokenizer(

        num_words=MAX_WORDS,

        oov_token="<OOV>"

    )

    tokenizer.fit_on_texts(df["message"])

    sequences = tokenizer.texts_to_sequences(

        df["message"]

    )

    X = pad_sequences(

        sequences,

        maxlen=MAX_LEN,

        padding="post"

    )

    y = df["label"]

    print("\nInput Shape :", X.shape)

    print("Output Shape:", y.shape)

    # Save Tokenizer

    with open(TOKENIZER, "wb") as f:

        pickle.dump(tokenizer, f)

    # Train-Test Split

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42

    )

    # Build Model

    model = Sequential()

    model.add(

        Embedding(

            input_dim=MAX_WORDS,

            output_dim=128,

            input_length=MAX_LEN

        )

    )

    model.add(

        SimpleRNN(

            128

        )

    )

    model.add(

        Dense(

            32,

            activation="relu"

        )

    )

    model.add(

        Dense(

            1,

            activation="sigmoid"

        )

    )

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    print("\nModel Summary\n")

    model.summary()
    # ==========================================================
# Continue Training
# ==========================================================

    history = model.fit(

        X_train,
        y_train,

        validation_split=0.20,

        epochs=10,

        batch_size=32,

        verbose=1

    )

    # Save Model

    model.save(MODEL)

    print("\nModel Saved Successfully.")

    # Evaluate

    loss, accuracy = model.evaluate(

        X_test,
        y_test,
        verbose=0

    )

    print(f"\nAccuracy : {accuracy:.4f}")

    predictions = (

        model.predict(
            X_test,
            verbose=0
        ) > 0.5

    ).astype(int).reshape(-1)

    print("\nClassification Report\n")

    print(

        classification_report(

            y_test,

            predictions

        )

    )

    print("\nConfusion Matrix\n")

    print(

        confusion_matrix(

            y_test,

            predictions

        )

    )


# ==========================================================
# Predict SMS
# ==========================================================

def predict_sms(message):

    # Load Model

    model = load_model(MODEL)

    # Load Tokenizer

    with open(TOKENIZER, "rb") as f:

        tokenizer = pickle.load(f)

    # Clean Message

    message = clean_text(message)

    # Convert Text to Sequence

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(

        sequence,

        maxlen=MAX_LEN,

        padding="post"

    )

    # Predict Spam Probability

    spam_probability = float(

        model.predict(

            sequence,

            verbose=0

        )[0][0]

    )

    # Ham Probability

    ham_probability = 1 - spam_probability

    # Final Prediction

    if spam_probability >= 0.5:

        prediction = "🚨 Spam"

        confidence = spam_probability

    else:

        prediction = "✅ Ham"

        confidence = ham_probability

    return (

        prediction,

        confidence,

        spam_probability,

        ham_probability

    )


# ==========================================================
# Train Automatically
# ==========================================================

if not os.path.exists(MODEL):

    if TENSORFLOW_AVAILABLE:

        train_model()

    else:

        print(

            "TensorFlow is not installed."

        )
        # ==========================================================
# Streamlit UI
# ==========================================================

if st is not None:

    st.title("SMS Spam Detector")

    st.write("Many to One RNN Example")

    message = st.text_area(
        "Enter SMS Message"
    )

    if st.button("Predict"):

        if message.strip() == "":

            st.warning("Please enter an SMS message.")

        else:

            prediction, confidence, spam_prob, ham_prob = predict_sms(message)

            # ------------------------
            # Prediction
            # ------------------------

            if "Spam" in prediction:

                st.error(prediction)

            else:

                st.success(prediction)

            # ------------------------
            # Confidence
            # ------------------------

            st.write("### Confidence Level")

            st.progress(confidence)

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            # ------------------------
            # Probability
            # ------------------------

            st.write("---")

            st.write(
                f"🚨 Spam Probability : {spam_prob*100:.2f}%"
            )

            st.write(
                f"✅ Ham Probability : {ham_prob*100:.2f}%"
            )

else:

    print(
        "Streamlit is not installed."
    )