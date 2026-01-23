#!/usr/bin/env python3
"""
Dataset Downloader for Image Model Training
Downloads sample datasets to get started quickly
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil

def download_file(url, destination):
    """Download file with progress bar"""
    print(f"Downloading from {url}...")
    
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end='')
    
    urllib.request.urlretrieve(url, destination, progress)
    print("\n✓ Download complete!")


def download_cifar10():
    """Download CIFAR-10 dataset (60k 32x32 images)"""
    print("\n📦 Downloading CIFAR-10 dataset...")
    print("   - 60,000 images (32x32)")
    print("   - 10 classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks")
    print("   - ~170MB download\n")
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    
    download_file(url, filename)
    
    print("Extracting...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    print("Converting to images...")
    import pickle
    import numpy as np
    from PIL import Image
    
    os.makedirs("training_images", exist_ok=True)
    
    # Load and save training batches
    for batch_num in range(1, 6):
        with open(f'cifar-10-batches-py/data_batch_{batch_num}', 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data']
            labels = batch[b'labels']
            
            # Reshape and save
            for i, (img_data, label) in enumerate(zip(images, labels)):
                img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img)
                img.save(f'training_images/cifar10_batch{batch_num}_{i:04d}_class{label}.png')
    
    # Cleanup
    os.remove(filename)
    shutil.rmtree('cifar-10-batches-py')
    
    print(f"\n✓ Successfully extracted 50,000 training images to 'training_images/'")
    print("  Ready to train!")


def download_flowers():
    """Download Oxford Flowers dataset (8k images)"""
    print("\n🌸 Downloading Oxford Flowers dataset...")
    print("   - 8,189 flower images")
    print("   - 102 flower categories")
    print("   - ~330MB download\n")
    
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    filename = "102flowers.tgz"
    
    download_file(url, filename)
    
    print("Extracting...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    # Move to training_images
    if os.path.exists('training_images'):
        shutil.rmtree('training_images')
    shutil.move('jpg', 'training_images')
    
    os.remove(filename)
    
    print(f"\n✓ Successfully extracted 8,189 images to 'training_images/'")
    print("  Ready to train!")


def create_sample_dataset():
    """Create a tiny synthetic dataset for testing"""
    print("\n🎨 Creating sample synthetic dataset...")
    print("   - 100 synthetic images (32x32)")
    print("   - For testing code only")
    print("   - Won't produce good results\n")
    
    from PIL import Image
    import numpy as np
    
    os.makedirs("training_images", exist_ok=True)
    
    for i in range(100):
        # Create random colored image
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        # Add some structure (gradient)
        for j in range(32):
            img[j, :, 0] = int(255 * j / 32)  # Red gradient
        
        img = Image.fromarray(img)
        img.save(f'training_images/synthetic_{i:03d}.png')
    
    print("✓ Created 100 synthetic images in 'training_images/'")
    print("  These are just for testing - use real datasets for actual training!")


def check_existing_images():
    """Check if training_images already exists"""
    if os.path.exists('training_images'):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
            image_files.extend(list(Path('training_images').glob(ext)))
            image_files.extend(list(Path('training_images').glob(ext.upper())))
        
        if len(image_files) > 0:
            return len(image_files)
    return 0


def main():
    print("=" * 70)
    print("  Image Model Training - Dataset Downloader")
    print("=" * 70)
    
    existing = check_existing_images()
    if existing > 0:
        print(f"\n⚠️  Found {existing} existing images in 'training_images/'")
        response = input("   Overwrite? (y/n): ").lower()
        if response != 'y':
            print("   Keeping existing images. Exiting.")
            return
        shutil.rmtree('training_images')
    
    print("\nChoose a dataset to download:\n")
    print("1. CIFAR-10 (Recommended for beginners)")
    print("   - 50,000 images, 32x32")
    print("   - 10 object categories")
    print("   - ~170MB download")
    print("   - Best for learning\n")
    
    print("2. Oxford Flowers")
    print("   - 8,189 images, various sizes")
    print("   - 102 flower species")
    print("   - ~330MB download")
    print("   - Good variety\n")
    
    print("3. Synthetic Test Dataset")
    print("   - 100 random images")
    print("   - Just for testing code")
    print("   - Won't produce good results\n")
    
    print("4. Manual Setup")
    print("   - I'll add my own images")
    print("   - Just create the folder for me\n")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        download_cifar10()
    elif choice == '2':
        download_flowers()
    elif choice == '3':
        create_sample_dataset()
    elif choice == '4':
        os.makedirs('training_images', exist_ok=True)
        print("\n✓ Created 'training_images/' folder")
        print("  Add your images there and run the training script!")
    else:
        print("\n❌ Invalid choice. Exiting.")
        return
    
    print("\n" + "=" * 70)
    print("  Next Steps:")
    print("=" * 70)
    print("\n1. For pixel-based model (32x32, recommended):")
    print("   python pixel_model_train.py\n")
    print("2. For patch-based model (64x64, advanced):")
    print("   python image_model_train.py\n")
    print("3. Read the guide:")
    print("   cat SETUP_GUIDE.md\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted. Cleaning up...")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("   Try downloading datasets manually")
