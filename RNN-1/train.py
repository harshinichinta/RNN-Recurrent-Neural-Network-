import pickle
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --------------------------------------------------
# Project directories
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Download dataset
# --------------------------------------------------
print("Downloading dataset...")

dataset_path = kagglehub.dataset_download("manann/quotes-500k")

csv_file = Path(dataset_path) / "quotes.csv"

df = pd.read_csv(csv_file)

texts = df["quote"].dropna().astype(str)

print("Total Quotes:", len(texts))

# Use first 200 quotes
texts = texts[:200]

print("Training Quotes:", len(texts))

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------
total_words = 5000

tokenizer = Tokenizer(
    num_words=total_words,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

# --------------------------------------------------
# Create training sequences
# --------------------------------------------------
input_sequences = []

for line in texts:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[: i + 1])

print("Total Sequences:", len(input_sequences))

# --------------------------------------------------
# Padding
# --------------------------------------------------
max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# --------------------------------------------------
# Build Model
# --------------------------------------------------
model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=50
    ),

    LSTM(64),

    Dense(
        total_words,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Train
# --------------------------------------------------
model.fit(
    X,
    y,
    epochs=3,
    batch_size=32,
    verbose=1
)

# --------------------------------------------------
# Save model
# --------------------------------------------------
model.save(MODEL_DIR / "rnn_model.keras")

with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open(MODEL_DIR / "max_sequence_len.pkl", "wb") as f:
    pickle.dump(max_sequence_len, f)

print("\nTraining Complete!")
print(f"Model saved to: {MODEL_DIR}")