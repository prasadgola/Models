import os
import tiktoken


print(tiktoken.encoding_for_model("gpt-2").encode("Hello Universe"))
