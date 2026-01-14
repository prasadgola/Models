import torch

with open("k.txt", 'r', encoding='utf-8') as f:
    text = f.read()

torch.manual_seed(1337)

# all possible symbols model can see or emit
chars = sorted(list(set(text)))

# token --> id and id --> token
token_ids = { ch:i for i, ch in enumerate(chars) }
chars_ids = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [token_ids[c] for c in s]
decode = lambda l: ''.join([chars_ids[i] for i in l])

tensor_data = torch.tensor(encode(text), dtype=torch.long)

# avoid over fitting and memorization of entire dataset. Divide dataset into 2 sets: train and validation
train_data = tensor_data[:int(0.9 * len(tensor_data))]
val_data = tensor_data[int(0.9 * len(tensor_data)):]


# Convert data into batches to run on GPU. multiple batches run in parallel. basically create matrix
batch_size = 64 # 32 chunks parallelly
chunk_size = 256 # each chunk size is 8
vocab_size = len(chars)
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.has_mps else 'cpu'
torch.device(device)
n_embd = 384
n_head = 6
n_layer = 6
Dropout = 0.2
# create input and target matrix with random chunks from dataset
def batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - chunk_size, (batch_size,))
    x = torch.stack([data[i:i+chunk_size] for i in ix])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Neural layer but not transformer, it's BLM
class BigramLanguageModelLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embd) # create lookup table for each token ID
        self.position_embedding = torch.nn.Embedding(chunk_size, n_embd)
        self.blocks = torch.nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embd, eps=1e-6)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)  # Convert 2d idx matrix into 3d matrix with vocab_size dimensions picked up from lookup table
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)  # Convert 3d matrix into 2d matrix to match cross entropy library matrix dimensions
            targets_reshaped = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_reshaped, targets_reshaped)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond = idx[:, -chunk_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(chunk_size, chunk_size)))
        self.dropout = torch.nn.Dropout(Dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.ln = torch.nn.LayerNorm(n_embd, eps=1e-6)
        self.dropout = torch.nn.Dropout(Dropout)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.ln(x)
        x = self.dropout(x)
        return x

class FeedForward(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.LayerNorm(n_embd, eps=1e-6),
            torch.nn.Dropout(Dropout),
            torch.nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, n_embd//num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = torch.nn.LayerNorm(n_embd, eps=1e-6)
        self.ln2 = torch.nn.LayerNorm(n_embd, eps=1e-6)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

model = BigramLanguageModelLayer()
m = model.to(device)

xb, yb = batch(train_data)
logits, loss = m(xb, yb)


optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)


# comnpetion layer level
for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    optimizer.zero_grad()
    xb, yb = batch(train_data)
    logits, loss = m(xb, yb)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx = context, max_new_tokens=500)[0].tolist()))

# head_size = 16
# key = torch.nn.Linear(n_embd, head_size, bias=False)
# query = torch.nn.Linear(n_embd, head_size, bias=False)
# value = torch.nn.Linear(n_embd, head_size, bias=False)
# k = key(x)
# q = query(x)
# wei = q @ k.transpose(-2, -1) * head_size**-0.5

# tril = torch.tril(torch.ones((chunk_size, chunk_size)))
# wei = torch.zeros((chunk_size, chunk_size))
# wei = wei.masked_fill(tril == 0, float('-inf'))
# wei = torch.softmax(wei, dim=-1)

# v = value(x)
# out = wei @ v


