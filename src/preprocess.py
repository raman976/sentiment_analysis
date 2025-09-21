# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import re


# def clean_text(text):
#     if pd.isnull(text):
#         text = ""
#     text = str(text)
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     return text.lower().strip()


# def load_and_prepare_data(file_path):
#     df = pd.read_csv(file_path)
#     df['cleaned_text'] = df['text'].apply(clean_text)
#     ohe = LabelEncoder()
#     df['sentiment'] = ohe.fit_transform(df[['sentiment']])
#     X = df['cleaned_text']
#     Y = df['sentiment']
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)
#     return X_train, X_test, Y_train, Y_test


import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Clean text
def clean_text(text):
    if pd.isnull(text):
        text = ""
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()

# Load data and encode labels
def load_and_prepare_data(file_path, test_size=0.2):
    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['text'].apply(clean_text)
    le = LabelEncoder()
    df['sentiment'] = le.fit_transform(df['sentiment'])
    X = df['cleaned_text'].tolist()
    Y = df['sentiment'].tolist()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test, le

# BERT Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

