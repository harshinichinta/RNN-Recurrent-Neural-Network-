import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Project directories
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = load_model(MODEL_DIR / "rnn_model.keras")

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------
with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# --------------------------------------------------
# Load maximum sequence length
# --------------------------------------------------
with open(MODEL_DIR / "max_sequence_len.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)

# --------------------------------------------------
# Generate quote
# --------------------------------------------------
def generate_quote(seed_text, next_words=20):

    generated_text = seed_text.strip()

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([generated_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        prediction = model.predict(token_list, verbose=0)

        predicted_index = np.argmax(prediction, axis=-1)[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # Stop if no valid word is found
        if output_word == "":
            break

        generated_text += " " + output_word

    return generated_text

# --------------------------------------------------
# Test
# --------------------------------------------------
if __name__ == "__main__":

    prompt = input("Enter Prompt: ")

    result = generate_quote(
        prompt,
        next_words=20
    )

    print("\nGenerated Quote:\n")
    print(result)