import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -----------------------------
# Create models folder
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv("mnist_train.csv")

print(df.head())

# -----------------------------
# Split Features and Labels
# -----------------------------
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# -----------------------------
# Normalize Images
# -----------------------------
X = X.astype("float32") / 255.0

# Convert 784 pixels → 28x28 sequence
X = X.reshape(-1, 28, 28)

print("Input Shape:", X.shape)

# -----------------------------
# One-Hot Encode Labels
# -----------------------------
y = to_categorical(y, num_classes=10)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# -----------------------------
# Build RNN Model
# -----------------------------
model = Sequential()

model.add(
    SimpleRNN(
        128,
        input_shape=(28, 28),
        activation="tanh"
    )
)

model.add(Dense(64, activation="relu"))

model.add(Dense(10, activation="softmax"))

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------
print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Accuracy:", accuracy)

# -----------------------------
# Save Model
# -----------------------------
model.save("models/one_to_one_rnn_model.keras")

print("\nModel Saved Successfully!")
print("Location : models/one_to_one_rnn_model.keras")