# import os
import tiktoken
import torch

intial_state = {
    "tokens": 50257,    # Vocabulary size
    "token_dimension": 768,         # Embedding dimension
    "context_length": 1024, # Context length
    "drop_rate": 0.1,       # Dropout rate
    "heads": 12,          # Number of attention heads
    "layers": 12,         # Number of layers
    "qkv_bias": False       # Query-Key-Value bias
}

class model(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.embedding_matrix_TxD = torch.nn.Embedding(variables["tokens"], variables["token_dimension"])
        self.position_embedding_matrix_CxD = torch.nn.Embedding(variables["context_length"], variables["token_dimension"])
        self.dropout = torch.nn.Dropout(variables["drop_rate"])

        # transfomer
        self.transformer = torch.nn.Sequential(*[transformer(variables) for _ in range(variables["layers"])])

        # last layer
        self.normalization = torch.nn.LayerNorm(variables["token_dimension"])
        self.output_layer = torch.nn.Linear(variables["token_dimension"], variables["tokens"], bias=False)

        
    def forward(self, batch):
        batch_size, chunk_size = batch.shape
        token_embeddings = self.embedding_matrix_TxD(batch)
        position_embeddings = self.position_embedding_matrix_CxD(torch.arange(chunk_size, device=batch.device).unsqueeze(0).expand(batch_size, -1))

        x = token_embeddings + position_embeddings
        x = self.transformer(x)
        x = self.normalization(x)
        x = self.output_layer(x)
        return x

class transformer(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        
    def forward(self, x):
        return x


class DummyLayer(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        # placeholder for now
        
    def forward(self, x):
        return x
            
model = model(intial_state)


tokenizer = tiktoken.get_encoding("gpt2")
english1 = "Every effort moves you"
english2 = "Every day holds a"



token_ID_batch = []
token_ID_batch.append(torch.tensor(tokenizer.encode(english1)))
token_ID_batch.append(torch.tensor(tokenizer.encode(english2)))

batch = torch.stack(token_ID_batch, dim=0)
logits = model(batch)