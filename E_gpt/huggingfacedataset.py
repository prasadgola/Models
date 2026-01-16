from datasets import load_dataset
import os

print("Downloading English text data (target: 1GB)...")

# Stream the dataset to avoid memory issues
dataset = load_dataset("openwebtext", split="train", streaming=True)

output_file = 'english.txt'
target_size = 1_000_000_000  # 1GB in bytes
current_size = 0
batch_size = 1000
doc_count = 0

with open(output_file, 'w', encoding='utf-8') as f:
    batch = []
    
    for item in dataset:
        batch.append(item['text'])
        doc_count += 1
        
        # Write in batches
        if len(batch) >= batch_size:
            text_chunk = '\n\n'.join(batch)
            f.write(text_chunk + '\n\n')
            current_size += len(text_chunk.encode('utf-8'))
            batch = []
            
            # Progress update
            if doc_count % 10000 == 0:
                mb_written = current_size / 1_000_000
                print(f"Progress: {mb_written:.1f}MB / 1000MB ({doc_count} documents)")
            
            # Stop at 1GB
            if current_size >= target_size:
                print(f"\n✓ Reached target size!")
                break
    
    # Write any remaining data
    if batch and current_size < target_size:
        text_chunk = '\n\n'.join(batch)
        f.write(text_chunk)
        current_size += len(text_chunk.encode('utf-8'))

final_size_mb = os.path.getsize(output_file) / 1_000_000
print(f"\n✓ Complete!")
print(f"File: {output_file}")
print(f"Size: {final_size_mb:.1f}MB")
print(f"Documents: {doc_count}")
print(f"Location: {os.path.abspath(output_file)}")