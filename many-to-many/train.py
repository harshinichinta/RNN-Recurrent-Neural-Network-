import os
import urllib.request
import pickle

import numpy as np
import tensorflow as tf

# -----------------------------
# Download Dataset (if missing)
# -----------------------------
DATASET_DIR = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "shakespeare.txt")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

if not os.path.exists(DATASET_PATH):
    print("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
        DATASET_PATH
    )
    print("Download completed.")

# -----------------------------
# Load Dataset
# -----------------------------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total Characters : {len(text)}")

# -----------------------------
# Character Vocabulary
# -----------------------------
chars = sorted(set(text))

char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)

print("Vocabulary Size:", vocab_size)

# -----------------------------
# Encode Text
# -----------------------------
encoded_text = np.array([char_to_index[c] for c in text])

# -----------------------------
# Create Sequences
# -----------------------------
sequence_length = 100

X = []
Y = []

for i in range(len(encoded_text) - sequence_length):
    X.append(encoded_text[i:i + sequence_length])
    Y.append(encoded_text[i + 1:i + sequence_length + 1])

X = np.array(X)
Y = np.array(Y)

print("Input Shape :", X.shape)
print("Target Shape:", Y.shape)

# -----------------------------
# Reduce Dataset Size (Optional)
# -----------------------------
# Change this number based on your RAM.
MAX_SAMPLES = 20000

if len(X) > MAX_SAMPLES:
    X = X[:MAX_SAMPLES]
    Y = Y[:MAX_SAMPLES]
    print(f"Using first {MAX_SAMPLES} samples for training.")

# -----------------------------
# One-Hot Encoding
# -----------------------------
X = tf.keras.utils.to_categorical(X, num_classes=vocab_size)
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)

print("One-hot Input Shape :", X.shape)
print("One-hot Target Shape:", Y.shape)

# -----------------------------
# Build Many-to-Many RNN Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length, vocab_size)),

    tf.keras.layers.SimpleRNN(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.SimpleRNN(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dense(
        vocab_size,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    X,
    Y,
    epochs=10,
    batch_size=64
)

# -----------------------------
# Save Model
# -----------------------------
model.save("model.h5")

# Save character mappings
with open("tokenizer.pkl", "wb") as f:
    pickle.dump((char_to_index, index_to_char), f)

print("\nModel saved as model.h5")
print("Tokenizer saved as tokenizer.pkl")
print("Training Completed Successfully!")