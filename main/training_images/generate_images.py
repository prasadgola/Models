"""
Image Generation Script
Load trained model and generate new images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
from pathlib import Path

# Import model architectures (simplified versions)
device = 'mps' if torch.backends.mps.is_built() else 'cpu'

# ============ PIXEL MODEL UTILS ============

def token_to_rgb_pixel(token, color_levels=16):
    """Convert token back to RGB for pixel model"""
    b = token % color_levels
    g = (token // color_levels) % color_levels
    r = (token // (color_levels ** 2)) % color_levels
    
    r = r / (color_levels - 1)
    g = g / (color_levels - 1)
    b = b / (color_levels - 1)
    
    return r, g, b


def tokens_to_image_pixel(tokens, image_size=32, color_levels=16):
    """Convert tokens to image for pixel model"""
    img = torch.zeros(3, image_size, image_size)
    
    for idx, token in enumerate(tokens[:image_size*image_size]):
        i = idx // image_size
        j = idx % image_size
        r, g, b = token_to_rgb_pixel(token.item(), color_levels)
        img[:, i, j] = torch.tensor([r, g, b])
    
    return img


# ============ PATCH MODEL UTILS ============

def tokens_to_image_patch(tokens, patch_size=8, patches_per_side=8, vocab_size=512):
    """Convert tokens to image for patch model"""
    num_patches = patches_per_side ** 2
    tokens = tokens[:num_patches]
    
    patches = []
    for token in tokens:
        r = (token // ((vocab_size // 8) ** 2)) % (vocab_size // 8)
        g = (token // (vocab_size // 8)) % (vocab_size // 8)
        b = token % (vocab_size // 8)
        
        rgb = torch.tensor([r, g, b], dtype=torch.float32) / (vocab_size // 8 - 1)
        patch = rgb.view(3, 1, 1).expand(3, patch_size, patch_size)
        patches.append(patch)
    
    img = torch.zeros(3, patches_per_side * patch_size, patches_per_side * patch_size)
    idx = 0
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            if idx < len(patches):
                img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[idx]
                idx += 1
    
    return img


# ============ MODEL LOADING ============

class Head(nn.Module):
    def __init__(self, n_embd, head_size, seq_len):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_len):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, seq_len) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)
    
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
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, seq_len):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, seq_len)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ImageGenerationModel(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embd=256, n_head=8, n_layer=6):
        super().__init__()
        self.seq_len = seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, seq_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:] if idx.shape[1] > self.seq_len else idx
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============ GENERATION FUNCTIONS ============

def generate_pixel_images(model_path='pixel_model_weights.pth', 
                         num_images=10,
                         temperature=0.9,
                         output_dir='generated_images'):
    """Generate images using pixel model"""
    
    print(f"\n🎨 Generating {num_images} images with pixel model...")
    print(f"   Temperature: {temperature}")
    
    # Model settings
    image_size = 32
    color_levels = 16
    vocab_size = color_levels ** 3
    seq_len = image_size * image_size
    
    # Load model
    model = ImageGenerationModel(vocab_size, seq_len, n_embd=256, n_head=8, n_layer=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate
    with torch.no_grad():
        for i in range(num_images):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=seq_len-1, temperature=temperature)
            
            img = tokens_to_image_pixel(generated[0].cpu(), image_size, color_levels)
            output_path = f'{output_dir}/pixel_generated_{i:03d}.png'
            save_image(img, output_path)
            print(f"   [{i+1}/{num_images}] Saved {output_path}")
    
    print(f"\n✓ Generated {num_images} images in '{output_dir}/'")


def generate_patch_images(model_path='image_model_weights.pth',
                         num_images=10,
                         temperature=0.9,
                         output_dir='generated_images'):
    """Generate images using patch model"""
    
    print(f"\n🎨 Generating {num_images} images with patch model...")
    print(f"   Temperature: {temperature}")
    
    # Model settings
    image_size = 64
    patch_size = 8
    vocab_size = 512
    patches_per_side = image_size // patch_size
    seq_len = patches_per_side ** 2
    
    # Load model
    model = ImageGenerationModel(vocab_size, seq_len, n_embd=256, n_head=8, n_layer=8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate
    with torch.no_grad():
        for i in range(num_images):
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=seq_len-1, temperature=temperature)
            
            img = tokens_to_image_patch(generated[0].cpu(), patch_size, patches_per_side, vocab_size)
            output_path = f'{output_dir}/patch_generated_{i:03d}.png'
            save_image(img, output_path)
            print(f"   [{i+1}/{num_images}] Saved {output_path}")
    
    print(f"\n✓ Generated {num_images} images in '{output_dir}/'")


def main():
    parser = argparse.ArgumentParser(description='Generate images from trained models')
    parser.add_argument('--model', type=str, required=True, choices=['pixel', 'patch'],
                       help='Model type to use')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights (default: pixel_model_weights.pth or image_model_weights.pth)')
    parser.add_argument('--num', type=int, default=10,
                       help='Number of images to generate (default: 10)')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature (default: 0.9). Lower=more conservative, Higher=more diverse')
    parser.add_argument('--output', type=str, default='generated_images',
                       help='Output directory (default: generated_images)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Image Generation Tool")
    print("=" * 70)
    
    # Determine weights path
    if args.weights is None:
        if args.model == 'pixel':
            args.weights = 'pixel_model_weights.pth'
        else:
            args.weights = 'image_model_weights.pth'
    
    # Check if weights exist
    if not Path(args.weights).exists():
        print(f"\n❌ Error: Model weights not found at '{args.weights}'")
        print("   Train a model first!")
        return
    
    # Generate
    if args.model == 'pixel':
        generate_pixel_images(args.weights, args.num, args.temperature, args.output)
    else:
        generate_patch_images(args.weights, args.num, args.temperature, args.output)
    
    print("\n💡 Tips:")
    print("   - Lower temperature (0.5-0.7): More consistent images")
    print("   - Higher temperature (1.0-1.2): More diverse images")
    print("   - Generate more images to see variety: --num 50")


if __name__ == "__main__":
    main()
