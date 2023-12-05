import argparse

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding import (
    autoregressive_decoding,
    generate,
    speculative_decoding,
    staged_speculative_decoding,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="c4")
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


def get_dataset(dataset_name: str, num_samples: int):
    if "c4" in dataset_name:
        dataset = load_dataset(dataset_name, "en", split="validation", streaming=True)
        dataset = dataset.take(num_samples)
        dataset = Dataset.from_generator(lambda: dataset)
        return dataset["text"]
    elif "stack" in dataset_name:
        dataset = load_dataset(
            dataset_name, data_dir="data/python", split=f"train[:{num_samples}]"
        )
        return dataset["content"]
    else:
        raise NotImplementedError


def print_model_info(model):
    params = sum(p.numel() for p in model.parameters())
    params = f"{params / 1e9:.3f} B" if params > 1e9 else f"{params / 1e6:.0f} M"
    memory = f"{model.get_memory_footprint() / 1024 ** 3:.2f} GB"
    print(f"Model: {model.name_or_path}, Parameters: {params}, Memory: {memory}")


def main():
    args = parse_args()

    # Instantiate the tokenizer, model, and draft model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    # Load the dataset
    dataset = get_dataset(args.dataset, args.num_samples)

    ard_time = generate(
        model,
        tokenizer,
        dataset,
        args.temperature,
        decoding=autoregressive_decoding,
    )
    spd_time = generate(
        model,
        tokenizer,
        dataset,
        args.temperature,
        decoding=speculative_decoding,
        draft_model=draft_model,
    )
    ssd_time = generate(
        model,
        tokenizer,
        dataset,
        args.temperature,
        decoding=staged_speculative_decoding,
        draft_model=draft_model,
    )
    print(f"time/token: {ard_time:.3f} ms, {spd_time:.3f} ms, {ssd_time:.3f} ms")
    print(f"speculative decoding speedup: {ard_time / spd_time:.3f}")
    print(f"staged speculative decoding speedup: {ard_time / spd_time:.3f}")


if __name__ == "__main__":
    main()
