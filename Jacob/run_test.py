from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

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
class MetaphorClassifier(torch.nn.Module):
    def __init__(self, tfidf_size, num_targets):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_targets, 8)
        self.fc1 = torch.nn.Linear(tfidf_size + 8, 128)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)

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
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    TEST_CSV = DATA_DIR / "test_data.csv"
    test_df = pd.read_csv(TEST_CSV)
    test_df = clean_labels(test_df)
    test_df["target_word"] = test_df["metaphorID"].map(TARGET_WORDS)
    test_df["context"] = test_df.apply(lambda r: extract_context(r["text"], r["target_word"]), axis=1)

    # Load vectorizer & model
    vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.pkl"
    model_path = OUTPUT_DIR / "model.pth"
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    X_test_tfidf = vectorizer.transform(test_df["context"]).toarray()
    target_ids = torch.tensor(test_df["metaphorID"].astype(int).values, dtype=torch.long)
    y_test = torch.tensor(test_df["label"].values, dtype=torch.long)

    model = MetaphorClassifier(tfidf_size=X_test_tfidf.shape[1], num_targets=len(TARGET_WORDS))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Dataloader
    dataset = TensorDataset(torch.tensor(X_test_tfidf, dtype=torch.float32), target_ids, y_test)
    dataloader = DataLoader(dataset, batch_size=32)

    # Predictions
    preds = []
    with torch.no_grad():
        for batch_X, batch_targets, _ in dataloader:
            out = model(batch_X, batch_targets)
            _, pred = torch.max(out, 1)
            preds.extend(pred.tolist())

    # Metrics
    y_true = y_test.tolist()
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average='macro')
    recall = recall_score(y_true, preds, average='macro')
    f1 = f1_score(y_true, preds, average='macro')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1:.4f}")

    # Save predictions
    predictions_path = OUTPUT_DIR / "predictions.csv"
    test_df["predicted_label"] = preds
    test_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path.resolve()}")

if __name__ == "__main__":
    main()
