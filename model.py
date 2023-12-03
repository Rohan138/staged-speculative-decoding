import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"]  # , "gpt2-xl"]
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello, my name is Rohan", return_tensors="pt")
inputs = inputs.to("cuda")

for model_name in models:
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    outputs = model(**inputs).logits
    print(model_name, sum(p.numel() for p in model.parameters()))
    del model, outputs

breakpoint()
