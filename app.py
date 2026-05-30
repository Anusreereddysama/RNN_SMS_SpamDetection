import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Detection",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/spam_lstm_model.keras"
    )

model = load_model()

tokenizer = joblib.load(
    "models/tokenizer.pkl"
)

encoder = joblib.load(
    "models/label_encoder.pkl"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📱 Model Information")

st.sidebar.info("""
Model: LSTM

Architecture:
Embedding → LSTM →
Dropout → Dense

Dataset:
SMS Spam Collection

Accuracy:
98.65%

Task:
Spam Classification
""")

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📩 RNN/LSTM SMS Spam Detection System")

st.markdown(
"""
Detect whether an SMS message is Spam or Ham
using Deep Learning and Natural Language Processing.
"""
)

# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Messages",
    "5,572"
)

c2.metric(
    "Model Accuracy",
    "98.65%"
)

c3.metric(
    "Algorithm",
    "LSTM"
)

st.divider()

# --------------------------------------------------
# Sample Messages
# --------------------------------------------------

with st.expander("View Example Messages"):

    st.success(
        "Hey, are we still meeting today?"
    )

    st.error(
        "Congratulations! You won a free iPhone. Click now!"
    )

# --------------------------------------------------
# Input
# --------------------------------------------------

message = st.text_area(
    "Enter SMS Message",
    height=150
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Analyze Message"):

    if len(message.strip()) == 0:

        st.warning(
            "Please enter a message."
        )

    else:

        seq = tokenizer.texts_to_sequences(
            [message]
        )

        padded = pad_sequences(
            seq,
            maxlen=100
        )

        prediction = model.predict(
            padded,
            verbose=0
        )[0][0]

        spam_prob = float(prediction)

        ham_prob = 1 - spam_prob

        st.subheader(
            "Prediction Result"
        )

        if spam_prob > 0.5:

            st.error(
                "🚨 SPAM MESSAGE DETECTED"
            )

            prediction_label = "Spam"

        else:

            st.success(
                "✅ HAM MESSAGE DETECTED"
            )

            prediction_label = "Ham"

        st.divider()

        # --------------------------------------
        # Metrics
        # --------------------------------------

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Spam Probability",
            f"{spam_prob*100:.2f}%"
        )

        m2.metric(
            "Ham Probability",
            f"{ham_prob*100:.2f}%"
        )

        confidence = max(
            spam_prob,
            ham_prob
        )

        m3.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.divider()

        # --------------------------------------
        # Progress Bar
        # --------------------------------------

        st.subheader(
            "Spam Risk Score"
        )

        st.progress(
            float(spam_prob)
        )

        # --------------------------------------
        # NLP Stats
        # --------------------------------------

        st.subheader(
            "Message Statistics"
        )

        words = len(
            message.split()
        )

        chars = len(
            message
        )

        s1, s2 = st.columns(2)

        s1.metric(
            "Word Count",
            words
        )

        s2.metric(
            "Character Count",
            chars
        )

        st.divider()

        st.subheader(
            "Analysis Summary"
        )

        st.info(
            f"""
Prediction: {prediction_label}

Spam Probability: {spam_prob*100:.2f}%

Model Confidence: {confidence*100:.2f}%
"""
        )