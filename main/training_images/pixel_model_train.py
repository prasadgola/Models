"""
Simpler Pixel-Based Image Transformer
Each pixel is treated as a token - easier to understand and debug
Works well for small images (32x32 or 64x64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
from pathlib import Path

# ============ HYPERPARAMETERS ============
image_size = 32  # 32x32 images
batch_size = 16
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1

# Each pixel has 3 channels (RGB), each with 256 possible values
# We'll use a simpler approach: quantize each channel to 16 levels
# This gives us 16*16*16 = 4096 possible colors per pixel
color_levels = 16
vocab_size = color_levels ** 3  # 4096 color combinations

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 20

device = 'mps' if torch.backends.mps.is_built() else 'cpu'
print(f"Using device: {device}")

# ============ DATASET ============

class PixelDataset(Dataset):
    """Treats each pixel as a token"""
    
    def __init__(self, image_folder, image_size=32):
        self.image_folder = Path(image_folder)
        self.image_size = image_size
        self.seq_len = image_size * image_size  # Total pixels
        
        # Find all images
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
            self.image_files.extend(list(self.image_folder.glob(ext)))
            self.image_files.extend(list(self.image_folder.glob(ext.upper())))
        
        print(f"Found {len(self.image_files)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def rgb_to_token(self, r, g, b):
        """Convert RGB values to single token"""
        # Quantize each channel from [0, 1] to [0, color_levels-1]
        r = int(r * (color_levels - 1))
        g = int(g * (color_levels - 1))
        b = int(b * (color_levels - 1))
        
        # Combine into single token
        token = r * (color_levels ** 2) + g * color_levels + b
        return token
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # (3, H, W)
        
        # Convert to tokens (flatten and quantize)
        tokens = []
        for i in range(self.image_size):
            for j in range(self.image_size):
                r, g, b = image[:, i, j]
                token = self.rgb_to_token(r.item(), g.item(), b.item())
                tokens.append(token)
        
        return torch.tensor(tokens, dtype=torch.long)


# ============ MODEL ============

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', None)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        
        if self.tril is None or self.tril.shape[0] < T:
            self.register_buffer('tril', torch.tril(torch.ones(T, T, device=x.device)))
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
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
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PixelGPT(nn.Module):
    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
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
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:] if idx.shape[1] > self.seq_len else idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============ UTILITIES ============

def token_to_rgb(token):
    """Convert token back to RGB"""
    b = token % color_levels
    g = (token // color_levels) % color_levels
    r = (token // (color_levels ** 2)) % color_levels
    
    # Scale back to [0, 1]
    r = r / (color_levels - 1)
    g = g / (color_levels - 1)
    b = b / (color_levels - 1)
    
    return r, g, b


def tokens_to_image(tokens, image_size):
    """Convert token sequence back to image"""
    img = torch.zeros(3, image_size, image_size)
    
    for idx, token in enumerate(tokens[:image_size*image_size]):
        i = idx // image_size
        j = idx % image_size
        r, g, b = token_to_rgb(token.item())
        img[:, i, j] = torch.tensor([r, g, b])
    
    return img


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    out = {}
    model.eval()
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        dataloader_iter = iter(loader)
        
        for k in range(min(eval_iters, len(loader))):
            try:
                batch = next(dataloader_iter).to(device)
                x = batch[:, :-1]
                y = batch[:, 1:]
                _, loss = model(x, y)
                losses.append(loss.item())
            except StopIteration:
                break
        
        out[split] = sum(losses) / len(losses) if losses else 0
    
    model.train()
    return out


# ============ MAIN ============

def main():
    image_folder = "training_images"
    
    if not os.path.exists(image_folder):
        print(f"\n⚠️  Please create '{image_folder}' folder and add images!")
        print("\nQuick start:")
        print("1. mkdir training_images")
        print("2. Add 100+ images to that folder")
        print("3. Run this script again")
        return
    
    # Dataset
    dataset = PixelDataset(image_folder, image_size=image_size)
    
    if len(dataset) == 0:
        print(f"\n⚠️  No images found in '{image_folder}'!")
        return
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    seq_len = image_size * image_size
    model = PixelGPT(vocab_size, seq_len).to(device)
    
    print(f"\n{'='*60}")
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Image size: {image_size}x{image_size} ({seq_len} pixels)")
    print(f"Vocab size: {vocab_size} colors ({color_levels} levels per channel)")
    print(f"Training: {train_size} images, Validation: {val_size} images")
    print(f"{'='*60}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load checkpoint
    if os.path.exists('pixel_model_weights.pth'):
        try:
            model.load_state_dict(torch.load('pixel_model_weights.pth'))
            print("✓ Loaded existing weights\n")
        except:
            print("✗ Starting fresh\n")
    
    # Training
    train_iter = iter(train_loader)
    
    try:
        for step in range(max_iters):
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model, train_loader, val_loader)
                print(f"step {step:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                torch.save(model.state_dict(), 'pixel_model_weights.pth')
                
                # Generate sample
                if step > 0:
                    model.eval()
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    generated = model.generate(context, max_new_tokens=seq_len-1, temperature=0.9)
                    img = tokens_to_image(generated[0].cpu(), image_size)
                    
                    from torchvision.utils import save_image
                    save_image(img, f'pixel_gen_step_{step}.png')
                    print(f"  → Saved pixel_gen_step_{step}.png")
                    model.train()
            
            # Train step
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted at step {step}")
    
    finally:
        torch.save(model.state_dict(), 'pixel_model_weights.pth')
        print("\n✓ Saved pixel_model_weights.pth")
        
        # Generate finals
        model.eval()
        for i in range(5):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=seq_len-1, temperature=0.9)
            img = tokens_to_image(generated[0].cpu(), image_size)
            from torchvision.utils import save_image
            save_image(img, f'final_pixel_{i}.png')
        
        print("✓ Generated 5 samples (final_pixel_0.png to final_pixel_4.png)")


if __name__ == "__main__":
    main()