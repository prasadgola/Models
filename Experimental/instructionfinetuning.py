# Instruction Fine-tuning - FIXED VERSION
import torch
import torch.nn as nn
import os
from torch.nn import functional as F
import json

# hyperparameters - MATCH BASE MODEL
batch_size = 64
block_size = 256  # Match base model
max_iters = 2000
eval_interval = 50
learning_rate = 1e-4
device = 'mps' if torch.backends.mps.is_built() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

print("Step 1: Loading base vocabulary from english.txt...")
# Load the ORIGINAL vocabulary from english.txt
with open('english.txt', 'r', encoding='utf-8') as f:
    base_text = f.read()

# Create base vocabulary
chars = sorted(list(set(base_text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Define special tokens
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

# Add special token characters to vocabulary
special_chars = set(USER_TOKEN + ASSISTANT_TOKEN + END_TOKEN)
for char in special_chars:
    if char not in stoi:
        idx = len(stoi)
        stoi[char] = idx
        itos[idx] = char
        chars.append(char)

vocab_size = len(stoi)
print(f"Vocabulary size: {vocab_size} (base: {len(set(base_text))}, special: {len(special_chars)})")

# Encoding/decoding functions
encode = lambda s: [stoi.get(c, 0) for c in s]  # Unknown chars -> 0
decode = lambda l: ''.join([itos.get(i, '') for i in l])

print("\nStep 2: Loading and formatting instruction data...")
# Load instruction dataset
with open('alpaca_data.json', 'r', encoding='utf-8') as f:
    instruction_data = json.load(f)

# Format as conversations
formatted_data = []
for item in instruction_data[:5000]:  # Use first 5000 for faster training
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    output = item.get('output', '')
    
    user_text = instruction
    if input_text:
        user_text += "\n" + input_text
    
    conversation = f"{USER_TOKEN}{user_text}{ASSISTANT_TOKEN}{output}{END_TOKEN}"
    formatted_data.append(conversation)

text = "\n".join(formatted_data)
print(f"Total training text length: {len(text)} characters")

# Encode data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

# Get special token IDs
assistant_token_ids = encode(ASSISTANT_TOKEN)
end_token_id = encode(END_TOKEN)[0]
print(f"Assistant token IDs: {assistant_token_ids}")
print(f"End token ID: {end_token_id}")

def get_batch(split):
    """Generate batch with loss masking for instruction finetuning"""
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    
    # Create mask: only train on assistant responses
    mask = torch.zeros_like(y, dtype=torch.bool)
    
    for i in range(batch_size):
        seq = x[i].tolist()
        in_assistant = False
        j = 0
        
        while j < len(seq):
            # Check if we're at the start of assistant token
            if j <= len(seq) - len(assistant_token_ids):
                if seq[j:j+len(assistant_token_ids)] == assistant_token_ids:
                    in_assistant = True
                    # Skip past the assistant token itself
                    j += len(assistant_token_ids)
                    continue
            
            # Check if we hit end token
            if seq[j] == end_token_id:
                in_assistant = False
            
            # Mark positions to train on (corresponding y position)
            if in_assistant and j < len(seq):
                mask[i, j] = True
            
            j += 1
    
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask = get_batch(split)
            logits, loss = model(X, Y, mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ===== MODEL CLASSES =====

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
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
    """Simple linear layer followed by non-linearity"""
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
    """Transformer block: communication followed by computation"""
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

    def forward(self, idx, targets=None, mask=None):
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
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            
            if mask is not None:
                mask_flat = mask.view(B*T)
                if mask_flat.sum() > 0:
                    loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
                else:
                    # Return a zero loss that still has gradients
                    loss = (logits_flat * 0).sum()
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, stop_tokens=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if stop_tokens and idx_next.item() in stop_tokens:
                break
        
        return idx

# ===== MAIN TRAINING =====

print("\nStep 3: Initializing model...")
model = GPTLanguageModel()
m = model.to(device)

# Load base model weights
if os.path.exists('model_weights.pth'):
    try:
        m.load_state_dict(torch.load('model_weights.pth', map_location=device))
        print("✓ Successfully loaded base model weights")
    except RuntimeError as e:
        print(f"✗ Error loading weights: {e}")
        print("⚠ Training from scratch instead")
else:
    print("⚠ No base model found - training from scratch")

print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nStep 4: Starting training...")
# Debug: Check if masking works
debug_x, debug_y, debug_mask = get_batch('train')
print(f"Debug - Mask coverage: {debug_mask.sum().item()} / {debug_mask.numel()} tokens ({100*debug_mask.float().mean():.1f}%)")
print(f"Debug - Example batch has assistant tokens: {debug_mask.sum() > 0}")

try:
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb, mask = get_batch('train')
        logits, loss = model(xb, yb, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

except KeyboardInterrupt:
    print(f"\n✗ Training interrupted at step {iter}")
    losses = estimate_loss()
    print(f"Final: train {losses['train']:.4f}, val {losses['val']:.4f}")

finally:
    torch.save(m.state_dict(), 'instruction_finetuned_weights.pth')
    print("✓ Model saved to instruction_finetuned_weights.pth")

# ===== TESTING =====

def chat(prompt):
    """Chat with the instruction-tuned model"""
    formatted_prompt = f"{USER_TOKEN}{prompt}{ASSISTANT_TOKEN}"
    context = torch.tensor([encode(formatted_prompt)], dtype=torch.long, device=device)
    response = m.generate(context, max_new_tokens=200, stop_tokens=[end_token_id])
    full_text = decode(response[0].tolist())
    
    # Extract assistant response
    if ASSISTANT_TOKEN in full_text:
        assistant_part = full_text.split(ASSISTANT_TOKEN)[-1]
        if END_TOKEN in assistant_part:
            assistant_response = assistant_part.split(END_TOKEN)[0]
        else:
            assistant_response = assistant_part
    else:
        assistant_response = full_text
    
    return assistant_response

print("\n" + "="*50)
print("CHAT TEST")
print("="*50)

test_questions = [
    "What is the capital of France?",
    "Explain what Python is",
    "Write a haiku about coding"
]

for q in test_questions:
    print(f"\nQ: {q}")
    print(f"A: {chat(q)}")
    print("-"*50)