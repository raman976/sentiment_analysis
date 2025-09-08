import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re


def clean_text(text):
    if pd.isnull(text):
        text = ""
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['text'].apply(clean_text)
    ohe = LabelEncoder()
    df['sentiment'] = ohe.fit_transform(df[['sentiment']])
    X = df['cleaned_text']
    Y = df['sentiment']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)
    return X_train, X_test, Y_train, Y_test