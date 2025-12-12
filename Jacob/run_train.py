from pathlib import Path
import argparse
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

def extract_context(sentence, target_word, window_size):
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
    parser = argparse.ArgumentParser(description="Train Jacob metaphor model.")
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "train.csv",
        help="Path to training CSV (expects columns: metaphorID, label, text).",
    )
    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path(__file__).resolve().parent / "model.pth",
        help="Path to save trained model weights.",
    )
    parser.add_argument(
        "--vectorizer_out",
        type=Path,
        default=Path(__file__).resolve().parent / "tfidf_vectorizer.pkl",
        help="Path to save fitted TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Context window size around target word (in tokens).",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Max features for TF-IDF vectorizer.",
    )
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parents[1]   # project root
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.vectorizer_out.parent.mkdir(parents=True, exist_ok=True)

    # Load & clean
    df = pd.read_csv(args.train_csv)
    df = clean_labels(df)
    df["target_word"] = df["metaphorID"].map(TARGET_WORDS)
    df["context"] = df.apply(lambda r: extract_context(r["text"], r["target_word"], args.window_size), axis=1)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_tfidf = vectorizer.fit_transform(df["context"]).toarray()

    # Targets & labels
    target_ids = torch.tensor(df["metaphorID"].astype(int).values, dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    dataset = TensorDataset(torch.tensor(X_tfidf, dtype=torch.float32), target_ids, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = MetaphorClassifier(tfidf_size=X_tfidf.shape[1], num_targets=len(TARGET_WORDS))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_targets, batch_y in dataloader:
            optimizer.zero_grad()
            out = model(batch_X, batch_targets)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss={total_loss/len(dataloader):.4f}")

    # Save model & vectorizer
    torch.save(model.state_dict(), args.model_out)
    with open(args.vectorizer_out, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model saved to {args.model_out}")
    print(f"Vectorizer saved to {args.vectorizer_out}")


if __name__ == "__main__":
    main()
