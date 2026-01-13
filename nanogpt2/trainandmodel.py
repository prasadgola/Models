import torch

with open("input.txt", 'r', encoding='utf-8') as f:
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


# Convert data into batches to run on GPU. multiple batches run in parallel
batch_size = 4 # 4 chunks parallelly
chunk_size = 8 # each chunk size is 8

def batch(data):
    ix = torch.randint(len(data) - chunk_size, (batch_size,))
    x = torch.stack([data[i:i+chunk_size] for i in ix])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
    return x, y

xb, yb = batch(train_data)

class Head(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):

        logits = self.token_embedding(idx)

        return logits

m = Head(vocab_size=len(chars))

out = m(xb, yb)
print(out)