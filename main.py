import argparse
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_LEN = 256
GEN_LEN = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--draft-model", type=str)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device-map", type=str, default="auto")

    args = parser.parse_args()

    args.load_in_8bit = False
    args.load_in_4bit = False
    if args.dtype is not None:
        if args.dtype == "fp16" or args.dtype == "float16":
            args.dtype = torch.float16
        elif args.dtype == "fp32" or args.dtype == "float32":
            args.dtype = torch.float32
        elif args.dtype == "bf16" or args.dtype == "bfloat16":
            args.dtype = torch.bfloat16
        elif args.dtype == "int8":
            args.dtype = torch.float16
            args.load_in_8bit = True
        elif args.dtype == "fp4":
            args.dtype = None
            args.load_in_4bit = True

    return args


def run_prediction_loop(
    model, tokenizer, num_samples, temperature=None, assistant_model=None
):
    outputs = []
    gen_time = []
    num_tokens = []
    ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
    ds_iterator = iter(ds)

    desc = "original model" if assistant_model is None else f"speculative model"
    pbar = tqdm(range(num_samples), desc)
    for _ in pbar:
        next_data = next(ds_iterator)["content"]
        inputs = tokenizer(
            [next_data], return_tensors="pt", max_length=INPUT_LEN, truncation=True
        )
        inputs = inputs.to(model.device)

        if temperature is not None:
            kwargs = {"temperature": temperature, "do_sample": True}
        else:
            kwargs = {"do_sample": False}

        start = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=GEN_LEN,
                pad_token_id=model.generation_config.eos_token_id,
                assistant_model=assistant_model,
                **kwargs,
            )
        end = time.time()

        outputs.append(tokenizer.decode(gen_out[0]))
        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

    avg_time = sum(gen_time) / sum(num_tokens) * 1000
    return avg_time


def print_model_info(model):
    params = sum(p.numel() for p in model.parameters())
    params = f"{params / 1e9:.3f} B" if params > 1e9 else f"{params / 1e6:.0f} M"
    memory = f"{model.get_memory_footprint() / 1024 ** 3:.2f} GB"
    print(f"Model: {model.name_or_path}, Parameters: {params}, Memory: {memory}")


def main():
    args = parse_args()

    # Instantiate the tokenizer, model, and draft model
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )
    print_model_info(model)

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )
    print_model_info(draft_model)

    orig_time = run_prediction_loop(
        model, tokenizer, args.num_samples, args.temperature
    )
    spec_time = run_prediction_loop(
        model, tokenizer, args.num_samples, args.temperature, draft_model
    )
    print(f"Time per token (ms): Original: {orig_time:.3f}, Speculative: {spec_time:.3f}")
    print(f"Speedup: {orig_time / spec_time:.3f}")


if __name__ == "__main__":
    main()
