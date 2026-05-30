import pandas as pd
import os

os.makedirs(
    "data/processed",
    exist_ok=True
)

df = pd.read_csv(
    "data/raw/spam.csv",
    encoding="latin-1"
)

# Keep only useful columns
df = df[["v1","v2"]]

# Rename columns
df.columns = [
    "label",
    "message"
]

# Remove duplicates
df = df.drop_duplicates()

print(df.head())
print(df.info())
print(df["label"].value_counts())

df.to_csv(
    "data/processed/cleaned_spam.csv",
    index=False
)

print("Processed dataset saved!")