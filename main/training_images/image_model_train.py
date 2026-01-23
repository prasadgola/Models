"""
Image Generation Transformer for M4 Pro (48GB Unified Memory)
Training on image patches with autoregressive generation

This model:
- Converts images to patches (like ViT)
- Trains transformer to predict next patch
- Generates images autoregressively
- Optimized for MPS backend on Apple Silicon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
import numpy as np
from pathlib import Path

# ============ HYPERPARAMETERS ============
# Image settings
image_size = 64  # 64x64 images (start small for faster training)
patch_size = 8   # 8x8 patches -> 64 patches per image
channels = 3     # RGB

# Model settings
batch_size = 32
n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.1
vocab_size = 512  # Number of discrete patch "tokens" (quantization levels)

# Training settings
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 50

# Device
device = 'mps' if torch.backends.mps.is_built() else 'cpu'
print(f"Using device: {device}")

# ============ DATA PROCESSING ============

class ImagePatchDataset(Dataset):
    """
    Converts images into sequences of patches for autoregressive training
    Each patch is quantized into discrete tokens
    """
    def __init__(self, image_folder, image_size=64, patch_size=8, vocab_size=512):
        self.image_folder = Path(image_folder)
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side ** 2
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
            self.image_files.extend(list(self.image_folder.glob(ext)))
            self.image_files.extend(list(self.image_folder.glob(ext.upper())))
        
        print(f"Found {len(self.image_files)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0, 1]
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def quantize_patch(self, patch):
        """Quantize a patch to discrete token"""
        # patch: (C, patch_size, patch_size)
        # Average pool to single value per channel, then quantize
        patch_mean = patch.mean(dim=(1, 2))  # (3,) for RGB
        
        # Quantize to vocab_size levels
        # Convert [0, 1] to [0, vocab_size-1]
        quantized = (patch_mean * (self.vocab_size - 1)).long()
        
        # Combine RGB into single token (simple approach)
        token = quantized[0] * (self.vocab_size // 8) ** 2 + \
                quantized[1] * (self.vocab_size // 8) + \
                quantized[2]
        
        # Ensure within vocab range
        token = token.clamp(0, self.vocab_size - 1)
        
        return token
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # (3, H, W)
        
        # Extract patches
        patches = []
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                patch = image[:, 
                            i*self.patch_size:(i+1)*self.patch_size,
                            j*self.patch_size:(j+1)*self.patch_size]
                token = self.quantize_patch(patch)
                patches.append(token)
        
        patches = torch.stack(patches)  # (num_patches,)
        return patches


class SimplerImageDataset(Dataset):
    """
    Alternative: Treat each pixel as a token (simpler but more tokens)
    Use this if the patch approach is too complex
    """
    def __init__(self, image_folder, image_size=32):
        self.image_folder = Path(image_folder)
        self.image_size = image_size
        
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
            self.image_files.extend(list(self.image_folder.glob(ext)))
        
        print(f"Found {len(self.image_files)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # (3, H, W)
        
        # Quantize to 256 levels (8-bit per channel)
        image = (image * 255).long()
        
        # Flatten: (3, H, W) -> (3*H*W,)
        tokens = image.flatten()
        
        return tokens


# ============ MODEL ARCHITECTURE ============

class Head(nn.Module):
    """Single self-attention head"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Will register causal mask dynamically in forward
        self.register_buffer('tril', None)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        
        # Apply causal mask
        if self.tril is None or self.tril.shape[0] < T:
            self.register_buffer('tril', torch.tril(torch.ones(T, T, device=x.device)))
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # GELU works better than ReLU for images
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""
    
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


class ImageGPT(nn.Module):
    """
    Image generation transformer
    Predicts next patch/pixel given previous patches/pixels
    """
    
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
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
        
        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        # Transformer
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -self.max_seq_len:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============ TRAINING UTILITIES ============

def get_batch(dataloader_iter, dataloader):
    """Get a batch of data"""
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
    
    batch = batch.to(device)
    
    # Create input and target (shifted by 1)
    x = batch[:, :-1]
    y = batch[:, 1:]
    
    return x, y, dataloader_iter


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        dataloader_iter = iter(loader)
        
        for k in range(eval_iters):
            try:
                X, Y, dataloader_iter = get_batch(dataloader_iter, loader)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            except:
                break
        
        out[split] = losses.mean()
    
    model.train()
    return out


def tokens_to_image(tokens, patch_size=8, patches_per_side=8, vocab_size=512):
    """Convert tokens back to image (approximate reconstruction)"""
    # This is a simplified reconstruction
    # In practice, you'd train a decoder or use VQ-VAE
    
    num_patches = patches_per_side ** 2
    tokens = tokens[:num_patches]
    
    # Dequantize tokens to RGB values
    patches = []
    for token in tokens:
        # Reverse the quantization
        r = (token // ((vocab_size // 8) ** 2)) % (vocab_size // 8)
        g = (token // (vocab_size // 8)) % (vocab_size // 8)
        b = token % (vocab_size // 8)
        
        # Scale back to [0, 1]
        rgb = torch.tensor([r, g, b], dtype=torch.float32) / (vocab_size // 8 - 1)
        
        # Create patch
        patch = rgb.view(3, 1, 1).expand(3, patch_size, patch_size)
        patches.append(patch)
    
    # Reconstruct image
    img = torch.zeros(3, patches_per_side * patch_size, patches_per_side * patch_size)
    idx = 0
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            if idx < len(patches):
                img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[idx]
                idx += 1
    
    return img


# ============ MAIN TRAINING LOOP ============

def main():
    # Create dataset
    image_folder = "training_images"  # Put your images here
    
    if not os.path.exists(image_folder):
        print(f"\n⚠️  Please create '{image_folder}' folder and add images!")
        print("You can use any image dataset (faces, objects, landscapes, etc.)")
        print("\nFor testing, you can download datasets from:")
        print("- Kaggle: https://www.kaggle.com/datasets")
        print("- Hugging Face: https://huggingface.co/datasets")
        print("\nExample datasets to try:")
        print("- CelebA faces (small version)")
        print("- CIFAR-10")
        print("- Your own photos")
        return
    
    # Create dataset and split
    dataset = ImagePatchDataset(
        image_folder, 
        image_size=image_size, 
        patch_size=patch_size,
        vocab_size=vocab_size
    )
    
    if len(dataset) == 0:
        print(f"\n⚠️  No images found in '{image_folder}'!")
        return
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # MPS works better with 0 workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    max_seq_len = (image_size // patch_size) ** 2
    model = ImageGPT(vocab_size, max_seq_len)
    model = model.to(device)
    
    print(f"\n{'='*50}")
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Sequence length: {max_seq_len} patches")
    print(f"Training images: {train_size}, Validation: {val_size}")
    print(f"{'='*50}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load checkpoint if exists
    if os.path.exists('image_model_weights.pth'):
        try:
            model.load_state_dict(torch.load('image_model_weights.pth'))
            print("✓ Loaded existing weights - continuing training\n")
        except:
            print("✗ Could not load weights - starting fresh\n")
    
    # Training loop
    train_iter = iter(train_loader)
    
    try:
        for step in range(max_iters):
            # Evaluate
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model, train_loader, val_loader)
                print(f"step {step:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                # Save checkpoint
                torch.save(model.state_dict(), 'image_model_weights.pth')
                
                # Generate a sample image
                if step > 0:
                    model.eval()
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    generated = model.generate(context, max_new_tokens=max_seq_len-1, temperature=0.8)
                    img = tokens_to_image(
                        generated[0].cpu(), 
                        patch_size=patch_size, 
                        patches_per_side=image_size//patch_size,
                        vocab_size=vocab_size
                    )
                    
                    # Save generated image
                    from torchvision.utils import save_image
                    save_image(img, f'generated_step_{step}.png')
                    print(f"  → Saved generated_step_{step}.png")
                    model.train()
            
            # Training step
            xb, yb, train_iter = get_batch(train_iter, train_loader)
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted at step {step}")
        losses = estimate_loss(model, train_loader, val_loader)
        print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    finally:
        # Save final model
        torch.save(model.state_dict(), 'image_model_weights.pth')
        print("\n✓ Model weights saved to image_model_weights.pth")
        
        # Generate final samples
        print("\n📸 Generating final samples...")
        model.eval()
        for i in range(5):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=max_seq_len-1, temperature=0.8)
            img = tokens_to_image(
                generated[0].cpu(), 
                patch_size=patch_size, 
                patches_per_side=image_size//patch_size,
                vocab_size=vocab_size
            )
            from torchvision.utils import save_image
            save_image(img, f'final_sample_{i}.png')
        
        print("✓ Saved 5 final samples (final_sample_0.png to final_sample_4.png)")


if __name__ == "__main__":
    main()
