import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TARGET_WORDS = {
    0: "road",
    1: "candle",
    2: "light",
    3: "spice",
    4: "ride",
    5: "train",
    6: "boat"
}

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def clean_labels(dataframe):
    if "label" in dataframe.columns:
        dataframe["label"] = (
            dataframe["label"].astype(str).str.strip().str.upper().map({"TRUE": 1, "FALSE": 0})
        )
    return dataframe.dropna(subset=["text", "metaphorID"])

def extract_context(sentence, target_word, window_size=5):
    tokens = sentence.split()
    indices = [i for i, w in enumerate(tokens) if w.lower() == target_word.lower()]
    if not indices:
        return sentence  # fallback if target not found
    i = indices[0]
    start = max(0, i - window_size)
    end = min(len(tokens), i + window_size + 1)
    return " ".join(tokens[start:end])

# ---------------------------
# MODEL DEFINITION
# ---------------------------
class MetaphorClassifier(nn.Module):
    def __init__(self, tfidf_size, num_targets):
        super().__init__()
        self.embedding = nn.Embedding(num_targets, 8)
        self.fc1 = nn.Linear(tfidf_size + 8, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x_tfidf, x_target):
        emb = self.embedding(x_target)
        x = torch.cat([x_tfidf, emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    # Paths
    train_path = '/Users/jacoblindsey/Documents/Machine Learning Class 2025/Team Project/train_data.csv'
    test_path = '/Users/jacoblindsey/Documents/Machine Learning Class 2025/Team Project/test_data.csv'

    # Load data
    df = pd.read_csv(train_path)
    td = pd.read_csv(test_path)

    # Clean labels
    df = clean_labels(df)
    td = clean_labels(td)

    # Extract context
    df["target_word"] = df["metaphorID"].map(TARGET_WORDS)
    td["target_word"] = td["metaphorID"].map(TARGET_WORDS)
    df["context"] = df.apply(lambda r: extract_context(r["text"], r["target_word"]), axis=1)
    td["context"] = td.apply(lambda r: extract_context(r["text"], r["target_word"]), axis=1)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(df["context"]).toarray()
    X_test_tfidf = vectorizer.transform(td["context"]).toarray()

    # Target embeddings
    unique_targets = len(TARGET_WORDS)
    target_ids_train = torch.tensor(df["metaphorID"].astype(int).values, dtype=torch.long)
    target_ids_test = torch.tensor(td["metaphorID"].astype(int).values, dtype=torch.long)

    # Convert TF-IDF and labels to tensors
    X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
    y_train_tensor = torch.tensor(df["label"].values, dtype=torch.long)
    y_test_tensor = torch.tensor(td["label"].values, dtype=torch.long)

    # DataLoaders
    train_dataset = TensorDataset(X_train_tensor, target_ids_train, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, target_ids_test, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model, loss, optimizer
    model = MetaphorClassifier(tfidf_size=X_train_tfidf.shape[1], num_targets=unique_targets)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_tfidf, batch_targets, batch_labels in train_loader:
            optimizer.zero_grad()
            out = model(batch_tfidf, batch_targets)
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

    # Testing
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_tfidf, batch_targets, batch_labels in test_loader:
            out = model(batch_tfidf, batch_targets)
            _, pred = torch.max(out, 1)
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")

# ---------------------------
# RUN MAIN
# ---------------------------
if __name__ == "__main__":
    main()
