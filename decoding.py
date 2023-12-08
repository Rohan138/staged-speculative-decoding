import time
from typing import List

import torch
from tqdm import tqdm

INP_LENGTH = 256
GEN_LENGTH = 128
NUM_TOKENS = 3
DEPTH = 3


def autoregressive_decoding(inputs, model, draft_model, temperature):
    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        use_cache=True,
        **kwargs,
    )


def speculative_decoding(
    inputs, model, draft_model, temperature=None, num_tokens=NUM_TOKENS
):
    if temperature is not None:
        kwargs = {"temperature": temperature, "do_sample": True}
    else:
        kwargs = {"do_sample": False}
    draft_model.generation_config.num_assistant_tokens = num_tokens
    # See https://huggingface.co/blog/assisted-generation
    # if n_matches == int(num_assistant_tokens):
    #     num_assistant_tokens += 2.0
    # else:
    #     num_assistant_tokens = max(1.0, num_assistant_tokens - 1.0)
    draft_model.generation_config.num_assistant_tokens_schedule = None  # "heuristic"
    return model.generate(
        **inputs,
        max_new_tokens=GEN_LENGTH,
        pad_token_id=model.generation_config.eos_token_id,
        assistant_model=draft_model,
        **kwargs,
    )


class DecodingNode:
    def __init__(self, inputs, draft_model, temperature=None, num_tokens=NUM_TOKENS):
        self.inputs = inputs
        self.draft_model = draft_model
        self.temperature = temperature
        self.num_tokens = num_tokens
        self.children: List[DecodingTree] = []

    def get_next_inputs(self):
        outputs = self.draft_model(**self.inputs, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        if self.temperature is not None:
            next_token_logits = next_token_logits / self.temperature

        next_token_logits, next_token_ids = next_token_logits.topk(
            self.num_tokens, dim=-1
        )
        mask = self.inputs["attention_mask"]
        attention_mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
        past_key_values = outputs.past_key_values

        for idx in range(self.num_tokens):
            yield {
                "input_ids": next_token_ids[:, idx].unsqueeze(1),
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }


class DecodingTree:
    def __init__(
        self, inputs, draft_model, temperature=None, num_tokens=NUM_TOKENS, depth=DEPTH
    ):
        self.inputs = inputs
        self.draft_model = draft_model
        self.temperature = temperature
        self.num_tokens = num_tokens
        self.depth = depth
        self.root = self.build()

    def build(self, depth=None):
        if depth is None:
            depth = self.depth
        root = DecodingNode(
            self.inputs, self.draft_model, self.temperature, self.num_tokens
        )
        eos_token_id = self.draft_model.generation_config.eos_token_id
        if depth > 0 and not self.inputs["input_ids"].any() == eos_token_id:
            for next_input in root.get_next_inputs():
                tree = DecodingTree(
                    next_input,
                    self.draft_model,
                    self.temperature,
                    self.num_tokens,
                    depth - 1,
                )
                root.children.append(tree)
        return root

    def __del__(self):
        # Delete all children; otherwise will OOM!
        if hasattr(self, "root") and hasattr(self.root, "children"):
            for child in self.root.children:
                del child
        del self


def count_nodes(root):
    return 1 + sum(count_nodes(child.root) for child in root.children)


def count_paths(root):
    if not root.children:
        return 1
    return sum(count_paths(child.root) for child in root.children)


def get_paths(root: DecodingNode):
    # For paths; concatenate `input_ids` along each path in the tree;
    # but only return `attention_mask` and `past_key_values` from terminal nodes
    if not root.children:
        return [root.inputs]
    paths = []
    for child in root.children:
        child_paths = get_paths(child.root)
        for child_path in child_paths:
            child_path["input_ids"] = torch.cat(
                [root.inputs["input_ids"], child_path["input_ids"]],
                dim=-1,
            )
            paths.append(child_path)
    return paths


def staged_speculative_decoding(
    inputs, model, draft_model, temperature=None, num_tokens=NUM_TOKENS, depth=DEPTH
):
    # Generate initial kv cache from model
    model_outputs = model(**inputs, use_cache=True)
    total_length = model_outputs.logits.shape[1]
    model_past_key_values = model_outputs.past_key_values
    output_sequence = inputs["input_ids"]
    eos_token_id = model.generation_config.eos_token_id

    # Sample initial last token
    new_logits = model_outputs.logits[:, -1]
    if temperature is not None:
        new_logits = new_logits / temperature
        dist = torch.distributions.Categorical(logits=new_logits)
        last_token = dist.sample()
    else:
        last_token = new_logits.argmax(dim=-1)

    # Generate initial kv cache for draft model
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    draft_outputs = draft_model(
        input_ids=input_ids[:, :-1],
        attention_mask=attention_mask[:, :-1],
        use_cache=True,
    )
    draft_past_key_values = draft_outputs.past_key_values
    inputs = {
        "input_ids": input_ids[:, -1:],
        "attention_mask": attention_mask,
        "past_key_values": draft_past_key_values,
    }

    while True:
        # Build staged speculative decoding tree
        tree = DecodingTree(inputs, draft_model, temperature, num_tokens, depth)

        # Print tree statistics; uncomment to debug
        # print(f"Number of nodes: {count_nodes(tree.root)}")
        # print(f"Number of paths: {count_paths(tree.root)}")

        # depth-first search to get all paths
        paths = get_paths(tree.root)

        input_ids = torch.cat([path["input_ids"] for path in paths], dim=0)
        attention_mask = torch.cat([path["attention_mask"] for path in paths], dim=0)

        # drop first token from input_ids since that is the input to the draft model
        # and is already represented by the previous kv cache
        input_ids = input_ids[:, 1:]

        # Expand past_key_values to match input_ids
        expanded_past_key_values = []
        for layer in model_past_key_values:
            expanded_layer = []
            for item in layer:
                expanded_item = item.expand(input_ids.shape[0], -1, -1, -1)
                expanded_layer.append(expanded_item)
            expanded_past_key_values.append(expanded_layer)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": expanded_past_key_values,
        }
        model_outputs = model(**model_inputs, use_cache=True)

        new_logits = model_outputs.logits[:, -depth - 1 :]

        if temperature is not None:
            new_logits = new_logits / temperature
            dist = torch.distributions.Categorical(logits=new_logits)
            selected_tokens = dist.sample()
        else:
            selected_tokens = new_logits.argmax(dim=-1)

        # find valid indices where input token predicted by
        # draft model is equal to previous output of model
        check_first = input_ids[:, 0] == last_token
        check_first = check_first.unsqueeze(1)
        check_rest = input_ids[:, 1:] == selected_tokens[:, :-1]
        check_all = torch.cat([check_first, check_rest], dim=1)
        valid_indices = torch.cumsum(check_all, dim=1)[:, -1]

        # choose the best valid index and find number of tokens generated
        chosen_index = valid_indices.argmax().item()
        num_tokens_generated = valid_indices[chosen_index].item()

        # update output sequence
        output_sequence = torch.cat(
            [
                output_sequence,
                input_ids[chosen_index].unsqueeze(0),
            ],
            dim=1,
        )

        # update values for next iteration
        last_token = selected_tokens[chosen_index][-1].unsqueeze(0)
        input_idx = num_tokens_generated - 1
        inputs["input_ids"] = input_ids[chosen_index][
            input_idx : input_idx + 1
        ].unsqueeze(0)
        inputs["attention_mask"] = attention_mask[chosen_index].unsqueeze(0)
        inputs["past_key_values"] = paths[chosen_index]["past_key_values"]
        model_past_key_values = []
        for layer in model_outputs.past_key_values:
            chosen_layer = []
            for item in layer:
                chosen_item = item[chosen_index].unsqueeze(0)
                chosen_layer.append(chosen_item)
            model_past_key_values.append(chosen_layer)

        total_length += num_tokens_generated
        if last_token == eos_token_id or total_length >= INP_LENGTH + GEN_LENGTH:
            break

    return output_sequence[:, : max(output_sequence.shape[1], INP_LENGTH + GEN_LENGTH)]


def generate(
    model,
    tokenizer,
    dataset,
    temperature,
    decoding,
    draft_model=None,
):
    assert decoding in [
        autoregressive_decoding,
        speculative_decoding,
        staged_speculative_decoding,
    ], f"Unknown decoding method: {decoding}"
    assert (
        decoding is not autoregressive_decoding or draft_model is None
    ), "Cannot use draft model with autoregressive decoding"

    gen_time = []
    num_tokens = []

    for data in tqdm(dataset, desc=decoding.__name__):
        inputs = tokenizer(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=INP_LENGTH,
        )
        inputs = inputs.to(model.device)

        torch.cuda.synchronize()
        start = time.monotonic()
        gen_out = decoding(inputs, model, draft_model, temperature)
        torch.cuda.synchronize()
        end = time.monotonic()

        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

    avg_time = sum(gen_time) / sum(num_tokens) * 1000
    return avg_time
