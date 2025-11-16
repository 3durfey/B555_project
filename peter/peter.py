import pandas as pd
from pathlib import Path
from transformers import BertTokenizerFast, BertModel
import torch
from target_embedding import get_target_embedding

# Load the BERT tokenizer associated with the "bert-base-uncased" model.
# Use the fast tokenizer so we can request offset mappings.
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Load the actual BERT neural network weights for "bert-base-uncased".
# This model converts token IDs into *contextualized embeddings*.
bert = BertModel.from_pretrained("bert-base-uncased")

# Put the model into evaluation mode.
# This disables dropout and other training-time layers.
# During evaluation, BERT should NOT update weights or apply noise.
bert.eval()

def get_target_embedding(text, target_word):
    # Tokenize with offsets
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )

    offsets = encoded["offset_mapping"][0]
    input_ids = encoded["input_ids"]

    # Find token indices matching the target word span
    target_word = target_word.lower()
    token_indices = []

    for i, (start, end) in enumerate(offsets):
        if text[start:end].lower() == target_word:
            token_indices.append(i)

    if len(token_indices) == 0:
        return None  # Word not found after tokenization

    # Feed into BERT
    with torch.no_grad():
        outputs = bert(input_ids, attention_mask=encoded["attention_mask"])

    hidden = outputs.last_hidden_state[0]  # shape: (tokens, 768)

    # Average embeddings if word split into multiple sub word tokens
    return hidden[token_indices].mean(dim=0)


def main():
    # Load CSV relative to this file so running from project root still works.
    csv_path = Path(__file__).resolve().parent / "train.csv"
    df = pd.read_csv(csv_path)

    # 'label' column becomes either 1 or 0.
    df["label"] = df["label"].map({True: 1, False: 0})

    # Target words.
    target_words = {
        0: "road",
        1: "candle",
        2: "light",
        3: "spice",
        4: "ride",
        5: "train",
        6: "boat"
    }
    embeddings = []

    for i, row in df.iterrows():
        text = row["text"]
        target = target_words[row["metaphorID"]]

        embedding = get_target_embedding(text, target)
        embeddings.append(embedding)
    print("done")

if __name__ == "__main__":
    main()
