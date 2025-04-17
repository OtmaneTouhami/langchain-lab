import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "L'inteligence artificielle change le monde."

tokens = tokenizer.tokenize(text)
tokens_id = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", tokens_id)

model = AutoModel.from_pretrained("gpt2")

tokens_tensor = torch.tensor([tokens_id])

with torch.no_grad():
    embeddings = model.get_input_embeddings()(tokens_tensor)


print("Embeddings of the first token:", embeddings[0][0].numpy())
