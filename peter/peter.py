import pandas as pd
from pathlib import Path
from transformers import BertTokenizerFast, BertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test_data.csv"

TARGET_WORDS = {
    0: "road",
    1: "candle",
    2: "light",
    3: "spice",
    4: "ride",
    5: "train",
    6: "boat"
}

WINDOW_CHARS = 300

# ---------------------------------------------
# 1. Load BERT tokenizer and model
# ---------------------------------------------

# Use the fast tokenizer so we can request offset mappings (char positions).
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Load the pre-trained BERT model encoder only.
bert = BertModel.from_pretrained("bert-base-uncased")

# Put BERT into evaluation mode (no dropout, no weight updates).
bert.eval()


# ---------------------------------------------
# 2. Function: get contextual embedding of target word
# ---------------------------------------------
def _find_target_spans(text: str, target_word: str):
    """Return start/end spans for occurrences of the target word."""
    text_lower = text.lower()
    target_lower = target_word.lower()
    start = 0
    matches = []
    suffixes = ["", "s", "es", "'s", "’s", "ed", "ing"]
    while True:
        idx = text_lower.find(target_lower, start)
        if idx == -1:
            break
        before = text_lower[idx - 1] if idx > 0 else " "
        if before.isalpha():
            start = idx + 1
            continue
        end_idx = idx + len(target_lower)
        span = None
        for suffix in suffixes:
            suffix_end = end_idx + len(suffix)
            if suffix_end > len(text_lower):
                continue
            if text_lower[end_idx:suffix_end] != suffix:
                continue
            after_char = text_lower[suffix_end] if suffix_end < len(text_lower) else " "
            if suffix and suffix.endswith("s") and after_char == "'":
                suffix_end += 1
                after_char = text_lower[suffix_end] if suffix_end < len(text_lower) else " "
            if after_char.isalpha():
                continue
            span = (idx, suffix_end)
            break
        if span is not None:
            matches.append(span)
        start = idx + 1
    return matches


def _token_indices_for_target(text: str, target_word: str, offsets):
    """Return token indices whose spans overlap the target word."""
    matches = _find_target_spans(text, target_word)

    for span_start, span_end in matches:
        indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:
                continue  # special tokens like [CLS]/[SEP]
            if tok_end <= span_start or tok_start >= span_end:
                continue
            indices.append(i)
        if indices:
            return indices
    return []


def get_target_embedding(text, target_word):
    """
    Given a sentence (text) and a target word (target_word),
    return the BERT contextual embedding (768-dim tensor)
    for that word in this specific context.

    If the word cannot be located after tokenization, returns None.
    """

    spans = _find_target_spans(text, target_word)
    if not spans:
        return None
    span_start, span_end = spans[0]
    window_start = max(0, span_start - WINDOW_CHARS)
    window_end = min(len(text), span_end + WINDOW_CHARS)
    chunk = text[window_start:window_end]

    # Tokenize the chunk with offset mapping so we know which characters each token spans.
    encoded = tokenizer(
        chunk,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )

    offsets = encoded["offset_mapping"][0]   # (num_tokens, 2) start/end char indices
    input_ids = encoded["input_ids"]         # token IDs fed to BERT

    token_indices = _token_indices_for_target(chunk, target_word, offsets)

    # If we never find the target word, we bail out.
    if len(token_indices) == 0:
        return None

    # Run the token IDs through BERT to get contextual embeddings.
    with torch.no_grad():
        outputs = bert(input_ids, attention_mask=encoded["attention_mask"])

    # last_hidden_state: (batch_size=1, num_tokens, hidden_size=768)
    hidden = outputs.last_hidden_state[0]  # shape: (num_tokens, 768)

    # If the word is split into multiple subword tokens, average their embeddings.
    return hidden[token_indices].mean(dim=0)


def load_dataset(csv_path: Path, split_name: str) -> pd.DataFrame:
    """Load a CSV and normalize the label column to integers."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{split_name} file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["label"] = (
        df["label"]
        .astype(str)
        #.str.replace('"', "")
        #.str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
    )
    if df["label"].isna().any():
        raise ValueError(f"Found unmapped labels in {split_name} split.")
    return df


def compute_embeddings(df: pd.DataFrame, split_name: str):
    """Compute embeddings for every row and keep None placeholders."""
    embeddings = []
    missing = 0
    for idx, row in df.iterrows():
        target = TARGET_WORDS.get(row["metaphorID"])
        if target is None:
            embeddings.append(None)
            missing += 1
            continue
        emb = get_target_embedding(row["text"], target)
        if emb is None:
            print(f"[{split_name}] target '{target}' not found for row {idx} (metaphorID {row['metaphorID']}).")
            missing += 1
        embeddings.append(emb)
    print(f"{split_name}: embeddings computed ({len(df) - missing}/{len(df)} found).")
    return embeddings


def build_prototypes(df: pd.DataFrame, embeddings):
    """Average literal embeddings inside the training data per target word."""
    literal_bank = {wid: [] for wid in TARGET_WORDS.keys()}
    for idx, row in df.iterrows():
        emb = embeddings[idx]
        if emb is None or row["label"] != 0:
            continue
        literal_bank[row["metaphorID"]].append(emb)

    prototypes = {}
    for wid, vecs in literal_bank.items():
        if len(vecs) == 0:
            print(f"WARNING: No literal examples for word ID {wid} in training data.")
            continue
        prototypes[wid] = torch.stack(vecs).mean(dim=0)
    return prototypes


def build_distance_features(df: pd.DataFrame, embeddings, prototypes, split_name: str):
    """Turn embeddings into distance features and aligned labels."""
    feats = []
    labels = []
    for idx, row in df.iterrows():
        emb = embeddings[idx]
        wid = row["metaphorID"]
        proto = prototypes.get(wid)
        if emb is None or proto is None:
            continue
        dist = torch.norm(emb - proto).item()
        feats.append(dist)
        labels.append(row["label"])

    if not feats:
        raise ValueError(f"No usable samples in {split_name} split.")

    X = np.array(feats, dtype=float).reshape(-1, 1)
    y = np.array(labels, dtype=int)
    print(f"{split_name}: using {len(X)} samples (literal {np.sum(y==0)}, metaphor {np.sum(y==1)}).")
    return X, y


def evaluate_split(df: pd.DataFrame, embeddings, split_name: str, prototypes, clf, return_details: bool = False):
    """Loop through a split, predict labels, and report metrics.

    If return_details is True, include per-example indices and predictions.
    """
    y_true = []
    y_pred = []
    evaluated_indices = []
    skipped = 0
    for idx, row in df.iterrows():
        emb = embeddings[idx]
        if emb is None:
            skipped += 1
            continue
        proto = prototypes.get(row["metaphorID"])
        if proto is None:
            skipped += 1
            continue
        dist = torch.norm(emb - proto).item()
        pred_label = int(clf.predict([[dist]])[0])
        y_true.append(int(row["label"]))
        y_pred.append(pred_label)
        evaluated_indices.append(idx)

    if not y_true:
        raise ValueError(f"No evaluable samples in {split_name} split.")

    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(
        f"{split_name} metrics — Accuracy: {accuracy:.4f}, "
        f"Precision (macro): {precision:.4f}, Recall (macro): {recall:.4f}, "
        f"F1 (macro): {f1:.4f} (evaluated {len(y_true)}, skipped {skipped})."
    )
    metrics = (accuracy, precision, recall, f1)
    if return_details:
        return metrics, {
            "indices": evaluated_indices,
            "y_true": y_true,
            "y_pred": y_pred,
            "skipped": skipped,
        }
    return metrics


# ---------------------------------------------
# 3. Training pipeline (build prototypes + classifier)
# ---------------------------------------------
def main():
    """
    Train on data/train_data.csv, evaluate on data/test_data.csv,
    and print the test accuracy.
    """
    train_df = load_dataset(TRAIN_CSV, "train")
    test_df = load_dataset(TEST_CSV, "test")

    train_embeddings = compute_embeddings(train_df, "train")
    test_embeddings = compute_embeddings(test_df, "test")

    prototypes = build_prototypes(train_df, train_embeddings)

    train_X, train_y = build_distance_features(train_df, train_embeddings, prototypes, "train")

    clf = LogisticRegression()
    clf.fit(train_X, train_y)

    evaluate_split(test_df, test_embeddings, "test", prototypes, clf)


if __name__ == "__main__":
    main()
