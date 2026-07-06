#RNN Practical
#SMS Spam Detection using Simple RNN(Many-to-One)
#Dataset:spam.csv
#import Libraries
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
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
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
#Configuration
MODEL="spam_model.keras"
TOKENIZER="tokenizer.pkl"

MAX_WORDS=5000
MAX_LEN=50
#CLEAN TEXT
def clean_text(text):
    text=str(text).lower()
    text=re.sub(r"[^a-z0-9]"," ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

#Train Model
def train_model():
    print("Training Dataset...")
    df = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0,1])
    df.columns = ["label", "message"]
    print(df.head())
    print(df["label"].value_counts())
    #convert labels into numbers
    df["label"]=df["label"].map({"ham":0,"spam":1})
    # clean sms
    df["message"] = df["message"].apply(clean_text)
    # Tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(df["message"])
    sequences = tokenizer.texts_to_sequences(df["message"])
    x = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post"
    )
    y=df["label"]
    print("X shape:",x.shape)
    print("Y shape:",y.shape)
    # save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)
    # Train, Test, Split
    x_train,x_test,y_train,y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )
    #Build RNN Model
    model=Sequential()
    #Embedding layer
    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        )
    )
    #Simple RNN Layer
    model.add(
        SimpleRNN(
            128
        )
    )
    #Hidden Layer
    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    #output layer
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
    model.summary()
    #Train
    history=model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1,
    )
    #save the model
    model.save(MODEL)

    # Evauate
    loss,accuracy=model.evaluate(x_test,y_test)
    print("\nAccuracy:",accuracy)

    # prediction
    predictions = (model.predict(x_test) > 0.5).astype(int).reshape(-1)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


def predict_sms(message):
    model = load_model(MODEL)
    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)
    message = clean_text(message)
    sequence = tokenizer.texts_to_sequences([message])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
    probability = model.predict(sequence, verbose=0)[0][0]
    if probability > 0.5:
        return "Spam", probability
    else:
        return "HAM", probability
if not os.path.exists(MODEL):
    if TENSORFLOW_AVAILABLE:
        train_model()
    else:
        print("TensorFlow is not installed. To train or run the model, install TensorFlow and rerun.")

if st is not None:
    # streamlit UI
    st.title("SMS Spam Detector")
    st.write(" Many to One RNN Example")
    message = st.text_area("Enter SMS message")

    if st.button("Predict"):
        prediction, probability = predict_sms(message)
        st.success(prediction)
        st.write("Confidence:", round(probability * 100, 2), "%")
else:
    print("Streamlit is not installed. The model is ready — call `predict_sms(message)` to get predictions.")
