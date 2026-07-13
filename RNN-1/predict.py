import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("models/rnn_model.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/max_sequence_len.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)


def generate_quote(seed_text, next_words=20):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [seed_text]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        predicted = np.argmax(
            model.predict(token_list, verbose=0),
            axis=-1
        )[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


if __name__ == "__main__":

    prompt = input("Enter Prompt: ")

    result = generate_quote(
        prompt,
        next_words=20
    )

    print("\nGenerated Text:\n")
    print(result)