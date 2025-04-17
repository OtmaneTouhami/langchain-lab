from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Once upon a time there was a robot who"
tokens_tensor = tokenizer(prompt, return_tensors='pt').input_ids

with torch.no_grad():
    output_tockens = model.generate(
        tokens_tensor,
        max_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
    )

generated_text = tokenizer.decode(output_tockens[0], skip_special_tokens=True)
print(f"Generated text\n{generated_text}")