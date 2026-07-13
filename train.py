import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

os.makedirs("models", exist_ok=True)

with open(
    "data/harry_potter.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read().lower()

print("Creating tokenizer...")

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Vocabulary Size:", total_words)

input_sequences = []

print("Creating sequences...")

for line in text.split("."):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

# Limit dataset size to avoid RAM issues
input_sequences = input_sequences[:50000]

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
print("y Shape:", y.shape)

model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=64,
        input_length=max_sequence_len - 1
    ),

    LSTM(128),

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

history = model.fit(
    X,
    y,
    epochs=6,
    batch_size=128,
    verbose=1
)

model.save("models/lstm_text_generator.keras")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/max_len.pkl", "wb") as f:
    pickle.dump(max_sequence_len, f)

print("\nModel Saved Successfully!")