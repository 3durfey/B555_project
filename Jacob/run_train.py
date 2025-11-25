from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# ---------------------------
# Constants
# ---------------------------
TARGET_WORDS = {0: "road", 1: "candle", 2: "light", 3: "spice", 4: "ride", 5: "train", 6: "boat"}

# ---------------------------
# Cleaning functions
# ---------------------------
def clean_labels(df):
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.upper().map({"TRUE": 1, "FALSE": 0})
    return df.dropna(subset=["text", "metaphorID"])

def extract_context(sentence, target_word, window_size=5):
    tokens = sentence.split()
    indices = [i for i, w in enumerate(tokens) if w.lower() == target_word.lower()]
    if not indices:
        return sentence
    i = indices[0]
    start = max(0, i - window_size)
    end = min(len(tokens), i + window_size + 1)
    return " ".join(tokens[start:end])

# ---------------------------
# Model
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
# Main
# ---------------------------
def main():
    # Project directories
    BASE_DIR = Path(__file__).resolve().parents[1]   # project root
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    TRAIN_CSV = DATA_DIR / "train_data.csv"

    # Load & clean
    df = pd.read_csv(TRAIN_CSV)
    df = clean_labels(df)
    df["target_word"] = df["metaphorID"].map(TARGET_WORDS)
    df["context"] = df.apply(lambda r: extract_context(r["text"], r["target_word"]), axis=1)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(df["context"]).toarray()

    # Targets & labels
    target_ids = torch.tensor(df["metaphorID"].astype(int).values, dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    dataset = TensorDataset(torch.tensor(X_tfidf, dtype=torch.float32), target_ids, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = MetaphorClassifier(tfidf_size=X_tfidf.shape[1], num_targets=len(TARGET_WORDS))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_X, batch_targets, batch_y in dataloader:
            optimizer.zero_grad()
            out = model(batch_X, batch_targets)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10, Loss={total_loss/len(dataloader):.4f}")

    # Save model & vectorizer
    model_path = OUTPUT_DIR / "model.pth"
    vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.pkl"
    torch.save(model.state_dict(), model_path)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


if __name__ == "__main__":
    main()


