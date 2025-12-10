import torch
import torch.nn as nn
from transformers import AutoModel

class MetaphorFusionModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_pos_tags=20, num_dep_tags=50, 
                 pos_embed_dim=16, dep_embed_dim=16, hidden_dim=128, dropout_prob=0.1):
        """
        Args:
            model_name (str): Pre-trained transformer model name.
            num_pos_tags (int): Number of unique POS tags.
            num_dep_tags (int): Number of unique Dependency tags.
            pos_embed_dim (int): Dimension of POS embeddings.
            dep_embed_dim (int): Dimension of Dependency embeddings.
            hidden_dim (int): Hidden dimension for the MLP.
            dropout_prob (float): Dropout probability.
        """
        super(MetaphorFusionModel, self).__init__()
        
        # 1. Backbone: DistilBERT
        self.bert = AutoModel.from_pretrained(model_name)

        bert_hidden_size = self.bert.config.hidden_size # 768 for distilbert-base
        
        # 2. Linguistic Embeddings
        self.pos_embedding = nn.Embedding(num_pos_tags, pos_embed_dim)
        self.dep_embedding = nn.Embedding(num_dep_tags, dep_embed_dim)
        
        # 3. Fusion & Classifier
        # Input to MLP = BERT_embedding + POS_embedding + Dep_embedding
        fusion_dim = bert_hidden_size + pos_embed_dim + dep_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 2) # Binary classification: Metaphor (1) vs Literal (0)
        )
        
    def forward(self, input_ids, attention_mask, target_index, pos_id, dep_id):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            target_index: (batch_size) - Index of the target word in the sequence
            pos_id: (batch_size)
            dep_id: (batch_size)
        """
        # 1. BERT Forward Pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # 2. Extract Target Word Embedding

        batch_size = input_ids.size(0)
        # Create a batch index tensor [0, 1, ..., batch_size-1]
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        
        # Select the vectors
        target_embeddings = last_hidden_state[batch_indices, target_index, :] # (batch_size, hidden_size)
        
        # 3. Lookup Linguistic Embeddings
        pos_embeds = self.pos_embedding(pos_id) # (batch_size, pos_embed_dim)
        dep_embeds = self.dep_embedding(dep_id) # (batch_size, dep_embed_dim)
        
        # 4. Concatenate
        fused_vector = torch.cat((target_embeddings, pos_embeds, dep_embeds), dim=1) # (batch_size, fusion_dim)
        
        # 5. Classify
        logits = self.mlp(fused_vector) # (batch_size, 2)
        
        return logits
