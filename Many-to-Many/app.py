import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# Streamlit Page
# --------------------------------------------------

st.set_page_config(
    page_title="Many-to-Many RNN",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Many-to-Many Recurrent Neural Network")
st.write("Named Entity Recognition using Simple RNN")

# --------------------------------------------------
# File Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "ner_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "many_to_many_rnn.keras")

WORD2IDX_PATH = os.path.join(BASE_DIR, "word2idx.pkl")

TAG2IDX_PATH = os.path.join(BASE_DIR, "tag2idx.pkl")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():

    try:
        df = pd.read_csv(DATA_PATH, encoding="latin1")
    except Exception:
        df = pd.read_csv(DATA_PATH)

    # Forward-fill missing values (Pandas 3.x compatible)
    df.ffill(inplace=True)

    return df
df = load_dataset()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Build Sentences
# --------------------------------------------------

def build_sentences(data):

    grouped = data.groupby("Sentence #")

    sentences = []

    tags = []

    for _, group in grouped:

        word_list = list(group["Word"].values)

        tag_list = list(group["Tag"].values)

        sentences.append(word_list)

        tags.append(tag_list)

    return sentences, tags

sentences, sentence_tags = build_sentences(df)

# --------------------------------------------------
# Vocabulary
# --------------------------------------------------

words = sorted(list(set(df["Word"].values)))

tags = sorted(list(set(df["Tag"].values)))

word2idx = {w:i+2 for i,w in enumerate(words)}
word2idx["PAD"] = 0
word2idx["UNK"] = 1

idx2word = {i:w for w,i in word2idx.items()}

tag2idx = {t:i+1 for i,t in enumerate(tags)}
tag2idx["PAD"] = 0

idx2tag = {i:t for t,i in tag2idx.items()}

# --------------------------------------------------
# Save Dictionaries
# --------------------------------------------------

pickle.dump(word2idx, open(WORD2IDX_PATH,"wb"))
pickle.dump(tag2idx, open(TAG2IDX_PATH,"wb"))

# --------------------------------------------------
# Convert to Integer Sequences
# --------------------------------------------------

X = []

y = []

for sent, tag_seq in zip(sentences, sentence_tags):

    word_ids = []

    tag_ids = []

    for w, t in zip(sent, tag_seq):

        word_ids.append(word2idx.get(w,1))

        tag_ids.append(tag2idx[t])

    X.append(word_ids)

    y.append(tag_ids)

# --------------------------------------------------
# Padding
# --------------------------------------------------

MAX_LEN = max(len(i) for i in X)

X = pad_sequences(
    X,
    maxlen=MAX_LEN,
    padding="post",
    value=0
)

y = pad_sequences(
    y,
    maxlen=MAX_LEN,
    padding="post",
    value=0
)

NUM_TAGS = len(tag2idx)

y = np.array([
    to_categorical(i, num_classes=NUM_TAGS)
    for i in y
])

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.success("Dataset Prepared Successfully")

st.write("Training Samples :", len(X_train))
st.write("Testing Samples :", len(X_test))
st.write("Maximum Sentence Length :", MAX_LEN)
st.write("Vocabulary Size :", len(word2idx))
st.write("Number of Tags :", NUM_TAGS)
# --------------------------------------------------
# Build Many-to-Many RNN Model
# --------------------------------------------------

def build_model(vocab_size, num_tags, max_len):

    model = Sequential()

    model.add(
        Embedding(
    input_dim=vocab_size,
    output_dim=128,
    mask_zero=True
)
    )

    model.add(
        SimpleRNN(
            units=128,
            return_sequences=True
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                64,
                activation="relu"
            )
        )
    )

    model.add(
        TimeDistributed(
            Dense(
                num_tags,
                activation="softmax"
            )
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

if os.path.exists(MODEL_PATH):

    st.success("Saved model found.")

    model = load_model(MODEL_PATH)

else:

    st.warning("Training model... This may take a few minutes.")

    model = build_model(
        vocab_size=len(word2idx),
        num_tags=NUM_TAGS,
        max_len=MAX_LEN
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(MODEL_PATH)

    st.success("Model trained and saved successfully.")


# --------------------------------------------------
# Model Summary
# --------------------------------------------------

st.subheader("Model Summary")

summary_lines = []
model.summary(print_fn=lambda x: summary_lines.append(x))

for line in summary_lines:
    st.text(line)


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

st.subheader("Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Loss",
        f"{loss:.4f}"
    )

with col2:
    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )


# --------------------------------------------------
# Training Information
# --------------------------------------------------

st.subheader("Dataset Information")

info = {
    "Vocabulary Size": len(word2idx),
    "Number of Tags": NUM_TAGS,
    "Maximum Sentence Length": MAX_LEN,
    "Training Samples": len(X_train),
    "Testing Samples": len(X_test)
}

st.table(
    pd.DataFrame(
        info.items(),
        columns=["Parameter", "Value"]
    )
)
