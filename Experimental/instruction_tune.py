import torch
import torch.nn as nn
import os
from torch.nn import functional as F

# Same hyperparameters as base model
batch_size = 32       # Smaller for instruction tuning
block_size = 128      
max_iters = 1000      # Fewer iterations needed for fine-tuning
eval_interval = 50
learning_rate = 1e-4  # Lower learning rate for fine-tuning
device = 'mps' if torch.backends.mps.is_built() else 'cpu'
eval_iters = 20
n_embd = 192
n_head = 4
n_layer = 4
dropout = 0.1         # Less dropout for instruction tuning

torch.manual_seed(1337)

# Load instruction dataset
with open('capitals_instructions.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenization (same as base model)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Save vocab for inference
with open('vocab_capitals.txt', 'w') as f:
    f.write(''.join(chars))

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocab size: {vocab_size}")
print(f"Training samples: {len(train_data):,} characters")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, stop_token=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop at special token if provided
            if stop_token is not None and idx_next.item() == stop_token:
                break
                
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(f"\n{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

# Load base model weights if available
if os.path.exists('model_weights_small.pth'):
    try:
        # Load base model
        base_state = torch.load('model_weights_small.pth', map_location=device)
        
        # Only load matching layers (embeddings won't match due to different vocab)
        current_state = m.state_dict()
        matching_state = {k: v for k, v in base_state.items() 
                         if k in current_state and v.shape == current_state[k].shape}
        
        m.load_state_dict(matching_state, strict=False)
        print(f"Loaded {len(matching_state)} layers from base model")
        print("Fine-tuning for instruction following\n")
    except Exception as e:
        print(f"Could not load base model: {e}")
        print("Training from scratch\n")
else:
    print("No base model found, training from scratch\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

try:
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

except KeyboardInterrupt:
    print(f"\n\nTraining interrupted at step {iter}")
    losses = estimate_loss()
    print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

finally:
    torch.save(m.state_dict(), 'model_capitals.pth')
    print("\nModel saved to model_capitals.pth")

# Test the model
print("\n" + "="*60)
print("Testing instruction following:")
print("="*60)

test_questions = [
    "What is the capital of France?",
    "What is the capital of Japan?", 
    "Tell me the capital of Brazil",
]

for question in test_questions:
    prompt = f"<|instruction|>{question}<|response|>"
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate until we hit the end token
    generated = m.generate(context, max_new_tokens=50)[0].tolist()
    full_text = decode(generated)
    
    # Extract just the response part
    if "<|response|>" in full_text and "<|end|>" in full_text:
        response = full_text.split("<|response|>")[1].split("<|end|>")[0]
        print(f"\nQ: {question}")
        print(f"A: {response}")
    else:
        print(f"\nQ: {question}")
        print(f"A: {full_text[len(prompt):]}")
