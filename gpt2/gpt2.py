import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class model(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.embedding_matrix_TxD = torch.nn.Embedding(variables["tokens"], variables["token_dimension"])
        self.position_embedding_matrix_CxD = torch.nn.Embedding(variables["context_length"], variables["token_dimension"])
        self.dropout = torch.nn.Dropout(variables["drop_rate"])

        # for loop n number of times which are layers
        self.transformer = torch.nn.Sequential(*[transformer(variables) for _ in range(variables["layers"])])

        # last layer
        # normalize one last time
        self.normalization = torch.nn.LayerNorm(variables["token_dimension"])
        # output layer
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

# transformer architecture
class transformer(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(variables["token_dimension"], variables["token_dimension"], variables["context_length"], variables["drop_rate"], variables["heads"], variables["qkv_bias"])
        self.feed_forward = FeedForward(variables)
        self.layer_norm1 = LayerNorm(variables["token_dimension"])
        self.layer_norm2 = LayerNorm(variables["token_dimension"])
        self.drop_shotcut = torch.nn.Dropout(variables["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.multihead_attention(x)
        x = self.drop_shotcut(x)
        x = x + shortcut
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shotcut(x)
        x = x + shortcut
        return x

# Multi head attention mechanism
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) 

        return context_vec

# activation function
class FeedForward(torch.nn.Module):
    def __init__(self, variables):
        super().__init__()
        self.linear1 = torch.nn.Linear(variables["token_dimension"], variables["token_dimension"])
        self.gelu = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(variables["token_dimension"], variables["token_dimension"])
        
    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

# Normalize the values
class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# decoder back to english
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  
        probas = torch.softmax(logits, dim=-1)  
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


class Dataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=2, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = Dataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
