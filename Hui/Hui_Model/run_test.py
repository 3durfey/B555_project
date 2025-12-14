# run_test.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# CPU-safe message
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("WARNING: This model runs slowly on CPU only. Please be patient.")

# -----------------------------
# Target word mapping
# -----------------------------
ID2WORD = {
    0: "road",
    1: "candle",
    2: "light",
    3: "spice",
    4: "ride",
    5: "train",
    6: "boat",
}

# -----------------------------
# Tokenizer
# -----------------------------
BERT_NAME = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)

# -----------------------------
# Encoding
# -----------------------------
def encode_with_targets(text, target_word, max_len=128):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len
    )

    input_ids = encoded["input_ids"]
    attn_mask = encoded["attention_mask"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    target_tokens = tokenizer.tokenize(target_word)
    L = len(target_tokens)

    target_mask = [0] * len(input_ids)
    for i in range(len(tokens) - L + 1):
        if tokens[i:i + L] == target_tokens:
            for j in range(L):
                target_mask[i + j] = 1

    return (
        torch.tensor(input_ids),
        torch.tensor(attn_mask),
        torch.tensor(target_mask),
    )

# -----------------------------
# Dataset
# -----------------------------
class MetaphorDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.target_ids = df["metaphorID"].astype(int).tolist()

    def __getitem__(self, idx):
        text = self.texts[idx]
        target_word = ID2WORD[self.target_ids[idx]]
        input_ids, attn_mask, target_mask = encode_with_targets(text, target_word)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "target_mask": target_mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.texts)

# -----------------------------
# Model
# -----------------------------
class MetaphorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME)
        self.target_emb = nn.Embedding(2, 20)

        self.fc1 = nn.Linear(768 + 20, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, target_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.pooler_output

        target_embed = self.target_emb(target_mask)
        pooled_target = target_embed.mean(dim=1)

        x = torch.cat([cls_embed, pooled_target], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

# -----------------------------
# Testing
# -----------------------------
def test(test_file):
    df = pd.read_csv(test_file)
    df["label"] = df["label"].astype(int)

    dataset = MetaphorDataset(df)
    loader = DataLoader(dataset, batch_size=8)

    model = MetaphorClassifier().to(device)
    model.load_state_dict(torch.load("model_Hui.pt", map_location=device))
    model.eval()

    preds, gts = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["target_mask"].to(device),
            )
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).long().squeeze(-1)

            preds.extend(pred.cpu().tolist())
            gts.extend(batch["label"].long().tolist())

    df["prediction"] = preds
    df.to_csv("predictions.csv", index=False)

    acc = accuracy_score(gts, preds)
    prec = precision_score(gts, preds, average="macro")
    rec = recall_score(gts, preds, average="macro")
    f1 = f1_score(gts, preds, average="macro")

    print("\nEvaluation Results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 (macro): {f1:.4f}")

if __name__ == "__main__":
    test("test_data.csv")
