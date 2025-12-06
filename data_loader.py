import torch
from torch.utils.data import Dataset
import pandas as pd
import spacy
from transformers import AutoTokenizer
import re

class MetaphorDataset(Dataset):
    def __init__(self, data_path, model_name='distilbert-base-uncased', max_len=128):
        """
        Args:
            data_path (str): Path to the CSV/JSON file containing the data.
            model_name (str): Name of the transformer model to use for tokenization.
            max_len (int): Maximum length of the tokenized sequence.
        """
        # Load data
        if data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        else:
            self.data = pd.read_csv(data_path)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        
        # Load spaCy model for linguistic feature extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm' model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Metaphor ID mapping to candidate words
        self.id_to_word = {
            0: 'road',
            1: 'candle',
            2: 'light',
            3: 'spice',
            4: 'ride',
            5: 'train',
            6: 'boat'
        }

        self.pos_to_idx = {
            'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CONJ': 4, 'CCONJ': 5, 
            'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 
            'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, 'SPACE': 18
        }
        
        self.dep_to_idx = {
            'acl': 0, 'acomp': 1, 'advcl': 2, 'advmod': 3, 'agent': 4, 'amod': 5, 'appos': 6, 
            'attr': 7, 'aux': 8, 'auxpass': 9, 'case': 10, 'cc': 11, 'ccomp': 12, 'compound': 13, 
            'conj': 14, 'csubj': 15, 'csubjpass': 16, 'dative': 17, 'dep': 18, 'det': 19, 'dobj': 20, 
            'expl': 21, 'intj': 22, 'mark': 23, 'meta': 24, 'neg': 25, 'nmod': 26, 'npadvmod': 27, 
            'nsubj': 28, 'nsubjpass': 29, 'nummod': 30, 'oprd': 31, 'parataxis': 32, 'pcomp': 33, 
            'pobj': 34, 'poss': 35, 'preconj': 36, 'predet': 37, 'prep': 38, 'prt': 39, 'punct': 40, 
            'quantmod': 41, 'relcl': 42, 'root': 43, 'xcomp': 44
        }
        # Add an UNK token for any unseen tags
        self.unk_idx = 99

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        metaphor_id = row['metaphorID']
        label = row['label']
        
        target_word = self.id_to_word[metaphor_id]
        
        # 1. Tokenize text with BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # 2. Find the index of the first occurrence of the target word in the BERT tokens

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        target_index = -1
        

        pattern = r'\b' + re.escape(target_word) + r'\b'
        match = re.search(pattern, text.lower())
        
        if match:
            start_char_idx = match.start()

            token_idx = encoding.char_to_token(0, start_char_idx)
            
            # If the start char maps to None (e.g., whitespace), try the next char
            if token_idx is None:
                 for i in range(len(target_word)):
                     token_idx = encoding.char_to_token(0, start_char_idx + i)
                     if token_idx is not None:
                         break
            
            if token_idx is not None:
                target_index = token_idx
            else:
                # Fallback: if mapping fails (truncation?), use CLS token (0) or handle error
                target_index = 0 
        else:
            # Word not found (should not happen in valid dataset)
            target_index = 0
            
        # 3. Extract Linguistic Features with spaCy
        doc = self.nlp(text)

        spacy_token = None
        for token in doc:
            if token.text.lower() == target_word:
                spacy_token = token
                break # First occurrence
        
        if spacy_token:
            pos_tag = spacy_token.pos_
            dep_tag = spacy_token.dep_
        else:
            # Fallback if spaCy tokenization differs significantly or word not found
            pos_tag = 'NOUN' # Default guess
            dep_tag = 'dobj' # Default guess
            
        pos_id = self.pos_to_idx.get(pos_tag, self.unk_idx)
        dep_id = self.dep_to_idx.get(dep_tag, self.unk_idx)

        if pos_id == self.unk_idx: pos_id = 0
        if dep_id == self.unk_idx: dep_id = 0

        # Handle label mapping if it's a string (TRUE/FALSE)
        if isinstance(label, str):
            if label.upper() == 'TRUE':
                label = 1
            elif label.upper() == 'FALSE':
                label = 0
            else:
                # Try to cast to int if it's a string number
                try:
                    label = int(label)
                except ValueError:
                    label = 0 # Default fallback
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_index': torch.tensor(target_index, dtype=torch.long),
            'pos_id': torch.tensor(pos_id, dtype=torch.long),
            'dep_id': torch.tensor(dep_id, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
