# RNN/LSTM SMS Spam Detection System

## Overview

This project implements an RNN-based SMS Spam Detection System using a Long Short-Term Memory (LSTM) network to classify text messages as Spam or Ham (Not Spam). The model leverages Natural Language Processing (NLP) techniques to understand message content and provide accurate predictions.

An interactive Streamlit dashboard allows users to enter SMS messages, analyze spam probability, view confidence scores, and receive real-time predictions.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- NLP

---

## Project Objectives

- Detect spam messages using deep learning
- Apply Natural Language Processing techniques
- Build a real-time text classification system
- Provide confidence-based prediction analysis
- Deploy an interactive web application

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Tokenization
5. Sequence Padding
6. LSTM Model Training
7. Model Evaluation
8. Streamlit Deployment

---

## Dataset

Dataset Used:

SMS Spam Collection Dataset

Features:

- SMS Message Text

Target Classes:

- Ham (Legitimate Message)
- Spam (Unwanted Message)

Dataset Statistics:

- Total Messages: 5,572
- Binary Classification Problem

---

## NLP Preprocessing

The following preprocessing techniques were applied:

- Removal of unnecessary columns
- Duplicate removal
- Label Encoding
- Tokenization
- Sequence Conversion
- Sequence Padding

---

## Model Architecture

```text
Input Message
      ↓
Tokenizer
      ↓
Padding
      ↓
Embedding Layer
      ↓
LSTM Layer
      ↓
Dropout Layer
      ↓
Dense Layer
      ↓
Spam / Ham
```

### Layers Used

- Embedding Layer
- LSTM Layer
- Dropout Layer
- Dense Output Layer

---

## Model Performance

### Evaluation Metric

- Accuracy Score

### Results

- Accuracy: **98.65%**

---

## Project Structure

```text
RNN_SMS_Spam_Detection/
│
├── app.py
├── preprocess.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── spam.csv
│   │
│   └── processed/
│       └── cleaned_spam.csv
│
├── models/
│   ├── spam_lstm_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
└── images/
```

---

## Dashboard Features

### Model Information

- LSTM Architecture Details
- Dataset Information
- Model Accuracy Display

### Message Analysis

- SMS Text Input
- Real-Time Spam Prediction
- Spam Probability Score
- Ham Probability Score
- Confidence Score

### NLP Statistics

- Word Count
- Character Count
- Spam Risk Indicator

### Interactive Visualization

- Progress-Based Spam Risk Score
- Prediction Summary Dashboard

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Anusreereddysama/RNN_SMS_Spam_Detection.git
```

Move into the project directory:

```bash
cd RNN_SMS_Spam_Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

Run:

```bash
python preprocess.py
```

This generates:

```text
data/processed/cleaned_spam.csv
```

---

## Model Training

Run:

```bash
python train_model.py
```

Generated files:

```text
models/spam_lstm_model.keras
models/tokenizer.pkl
models/label_encoder.pkl
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

## Example Messages

### Ham Message

```text
Hey, are we meeting today at 5 PM?
```

Prediction:

```text
Ham
```

### Spam Message

```text
Congratulations! You have won a free iPhone. Claim now!
```

Prediction:

```text
Spam
```

---

## Features

- Deep Learning-Based Spam Detection
- LSTM Neural Network
- Natural Language Processing Pipeline
- Real-Time Message Classification
- Spam Probability Analysis
- Confidence Score Evaluation
- Interactive Streamlit Dashboard
- Clean and Modular Project Structure

---

## Future Enhancements

- Bidirectional LSTM
- GRU-Based Classification
- Attention Mechanism
- Multi-Class Text Classification
- Email Spam Detection
- Transformer-Based NLP Models
- Cloud Deployment

---
