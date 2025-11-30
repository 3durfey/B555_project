### This code read in the whole train.csv and perform 80/20 train-test split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


# ---------------------------------------------------------
# 1. Target word mapping from metaphorID
# ---------------------------------------------------------
ID2WORD = {
    0: "road",
    1: "candle",
    2: "light",
    3: "spice",
    4: "ride",
    5: "train",
    6: "boat",
}


# ---------------------------------------------------------
# 2. Load tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ---------------------------------------------------------
# 3. Encode sentence + target word marking
# ---------------------------------------------------------
def encode_with_targets(text, target_word, max_len=128):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_offsets_mapping=True
    )

    input_ids = encoded["input_ids"]
    attn_mask = encoded["attention_mask"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    target_tokens = tokenizer.tokenize(target_word)
    L = len(target_tokens)

    # Build target mask
    target_mask = [0] * len(input_ids)

    occurrences = 0
    for i in range(len(tokens) - L + 1):
        if tokens[i:i + L] == target_tokens:
            occurrences += 1
            for j in range(L):
                target_mask[i + j] = 1

    # Inform user if target word appears multiple times
    if occurrences > 1:
        print(f"Multiple target words detected ({occurrences}x): '{target_word}'")

    return (
        torch.tensor(input_ids),
        torch.tensor(attn_mask),
        torch.tensor(target_mask),
    )


# ---------------------------------------------------------
# 4. Dataset
# ---------------------------------------------------------
class MetaphorDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()  # True/False → 1/0
        self.target_ids = df["metaphorID"].astype(int).tolist()

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        target_word = ID2WORD[self.target_ids[idx]]

        input_ids, attn_mask, target_mask = encode_with_targets(text, target_word)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "target_mask": target_mask,
            "label": label
        }

    def __len__(self):
        return len(self.texts)


# ---------------------------------------------------------
# 5. Model definition
# ---------------------------------------------------------
class MetaphorClassifier(nn.Module):
    def __init__(self):
        super(MetaphorClassifier, self).__init__()
        
        # BERT encoder
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # target mask embedding (0 or 1 → 20-dim vector)
        self.target_emb = nn.Embedding(2, 20)

        # input size now = 768 (CLS) + 20 (target embedding)
        self.fc1 = nn.Linear(768 + 20, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, target_mask):
        # BERT output
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        cls_embed = outputs.pooler_output         # shape: (B, 768)
        target_embed = self.target_emb(target_mask)  # shape: (B, L, 20)

        # mean-pool all target-token embeddings
        pooled_target = target_embed.mean(dim=1)  # shape: (B, 20)

        # concatenate CLS + target embedding
        x = torch.cat([cls_embed, pooled_target], dim=1)

        # your architecture
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


# ---------------------------------------------------------
# 6. Training function
# ---------------------------------------------------------
def train_model(df, epochs=5, batch_size=8, lr=2e-5):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = MetaphorDataset(train_df)
    val_ds = MetaphorDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MetaphorClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tmask = batch["target_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(ids, attn, tmask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                tmask = batch["target_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(ids, attn, tmask)
                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).long()

                preds.extend(pred.cpu().tolist())
                gts.extend(labels.long().cpu().tolist())

        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss = {total_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f}")

    return model

if __name__ == "__main__":
  df = pd.read_csv("train.csv")
  df["label"] = df["label"].astype(int)
  train_model(df)
