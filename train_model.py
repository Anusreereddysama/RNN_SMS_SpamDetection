import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

df = pd.read_csv(
    "data/processed/cleaned_spam.csv"
)

# Encode labels
encoder = LabelEncoder()

df["label"] = encoder.fit_transform(
    df["label"]
)

X = df["message"]

y = df["label"]

# Tokenization
max_words = 5000

tokenizer = Tokenizer(
    num_words=max_words
)

tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)

X_pad = pad_sequences(
    X_seq,
    maxlen=100
)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42
)

# LSTM Model

model = Sequential([

    Embedding(
        input_dim=max_words,
        output_dim=64,
        input_length=100
    ),

    LSTM(64),

    Dropout(0.2),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    f"Accuracy: {accuracy:.4f}"
)

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/spam_lstm_model.keras"
)

joblib.dump(
    tokenizer,
    "models/tokenizer.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("Model saved successfully!")