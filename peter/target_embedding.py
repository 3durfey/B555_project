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
