import os
import pickle
import kagglehub
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print("Downloading dataset...")

path = kagglehub.dataset_download("manann/quotes-500k")

csv_file = os.path.join(path, "quotes.csv")

df = pd.read_csv(csv_file)

texts = df["quote"].dropna().astype(str)

print("Total Quotes:", len(texts))

# Use only 200 quotes
texts = texts[:200]

print("Training Quotes:", len(texts))

tokenizer = Tokenizer(
    num_words=5000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

total_words = 5000

input_sequences = []

for line in texts:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(
            token_list[:i + 1]
        )

print("Total Sequences:", len(input_sequences))

max_sequence_len = max(
    len(seq)
    for seq in input_sequences
)

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

model.fit(
    X,
    y,
    epochs=3,
    batch_size=32,
    verbose=1
)

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/rnn_model.keras"
)

with open(
    "models/tokenizer.pkl",
    "wb"
) as f:
    pickle.dump(tokenizer, f)

with open(
    "models/max_sequence_len.pkl",
    "wb"
) as f:
    pickle.dump(max_sequence_len, f)

print("\nTraining Complete!")
print("Model saved successfully.")