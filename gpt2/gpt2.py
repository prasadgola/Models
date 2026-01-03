import os
import tiktoken


tokenizer = tiktoken.encoding_for_model("gpt-2")

with open("training_data.txt", "r", encoding="utf-8") as f:
    training_data = f.read()

# convert traning data to token ids - (encoder)
training_token_ids = tokenizer.encode(training_data, allowed_special={"<|space|>"})




