# import os
# import tiktoken
import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        # self.transformer = nn.ModuleList([nn.ModuleList([nn.Linear(config["emb_dim"], config["emb_dim"]), nn.Linear(config["emb_dim"], config["emb_dim"]), nn.Linear(config["emb_dim"], config["emb_dim"]), nn.Linear(config["emb_dim"], config["emb_dim"])]) for _ in range(config["n_layers"])]

        self.transformer_block = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"]),
            nn.Linear(config["emb_dim"], config["emb_dim"]),
            nn.Linear(config["emb_dim"], config["emb_dim"]),
            nn.Linear(config["emb_dim"], config["emb_dim"]),
        )

        self.final_normalization = nn.LayerNorm(config["emb_dim"])
        self.final_linear = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_length = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1))
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.transformer_block(x)
        x = self.final_normalization(x)
        x = self.final_linear(x)
        return x


class transformer_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # placeholder for now
        
    def forward(self, x):
        return x

class DummyLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # placeholder for now
        
    def forward(self, x):
        return x