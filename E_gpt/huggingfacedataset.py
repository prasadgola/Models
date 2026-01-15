from datasets import load_dataset

dataset = load_dataset("openwebtext", split="train")
text = '\n'.join([item['text'] for item in dataset])

with open('english.txt', 'w', encoding='utf-8') as f:
    f.write(text[:1_000_000_000])  # ~1GB