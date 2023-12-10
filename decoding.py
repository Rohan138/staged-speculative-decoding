import copy
import time
from typing import List

import torch
from torch.nn import functional as F
from tqdm import tqdm

INP_LENGTH = 256
GEN_LENGTH = 128
NUM_TOKENS = 5
TOPK = 3


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
    def __init__(self, inputs, draft_model, temperature=None, topk=TOPK):
        self.inputs = inputs
        self.draft_model = draft_model
        self.temperature = temperature
        self.topk = topk
        self.children: List[DecodingTree] = []

    def get_next_inputs(self):
        outputs = self.draft_model(**self.inputs, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        if self.temperature is not None:
            next_token_logits = next_token_logits / self.temperature

        next_token_logits, next_token_ids = next_token_logits.topk(self.topk, dim=-1)
        mask = self.inputs["attention_mask"]
        attention_mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
        past_key_values = outputs.past_key_values

        for idx in range(self.topk):
            yield {
                "input_ids": next_token_ids[:, idx : idx + 1],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }


class DecodingTree:
    def __init__(
        self,
        inputs,
        draft_model,
        temperature=None,
        topk=TOPK,
        num_tokens=NUM_TOKENS,
    ):
        self.inputs = inputs
        self.draft_model = draft_model
        self.temperature = temperature
        self.topk = topk
        self.num_tokens = num_tokens
        self.root = self.build()

    def build(self, num_tokens=None):
        if num_tokens is None:
            num_tokens = self.num_tokens
        root = DecodingNode(self.inputs, self.draft_model, self.temperature, self.topk)
        eos_token_id = self.draft_model.generation_config.eos_token_id
        if num_tokens > 0 and not (self.inputs["input_ids"] == eos_token_id).any():
            for next_input in root.get_next_inputs():
                tree = DecodingTree(
                    next_input,
                    self.draft_model,
                    self.temperature,
                    self.topk,
                    num_tokens - 1,
                )
                root.children.append(tree)
        return root


def count_nodes(root):
    return 1 + sum(count_nodes(child.root) for child in root.children)


def count_paths(root):
    if not root.children:
        return 1
    return sum(count_paths(child.root) for child in root.children)


def get_tree_input_ids(root: DecodingNode):
    # concatenate `input_ids` along each node
    root_input_ids = root.inputs["input_ids"]
    if not root.children:
        return [root_input_ids]
    input_ids = []
    for child in root.children:
        for child_input_ids in get_tree_input_ids(child.root):
            cat_input_ids = torch.cat([root_input_ids, child_input_ids], dim=-1)
            input_ids.append(cat_input_ids)
    return input_ids


def get_tree_past_key_values(root: DecodingNode):
    # concatenate `past_key_values` along each node
    root_past_key_values = root.inputs["past_key_values"]
    if not root.children:
        return [[root_past_key_values]]
    past_key_values = []
    for child in root.children:
        for child_past_key_values in get_tree_past_key_values(child.root):
            cat_past_key_values = [root_past_key_values, *child_past_key_values]
            past_key_values.append(cat_past_key_values)
    return past_key_values


@torch.no_grad()
def staged_speculative_decoding(
    inputs, model, draft_model, temperature=None, topk=TOPK, num_tokens=NUM_TOKENS
):
    inputs = copy.copy(inputs)
    output_sequence = inputs["input_ids"]
    total_length = output_sequence.shape[1]
    eos_token_id = model.generation_config.eos_token_id

    # Generate initial kv cache for model
    model_inputs = {
        "input_ids": inputs["input_ids"][:, :-1],
        "attention_mask": inputs["attention_mask"][:, :-1],
        "past_key_values": None,
    }
    model_outputs = model(**model_inputs, use_cache=True)
    prev_inputs = {}

    while True:
        # Generate initial kv cache for draft model
        # currently need to rerun this every iteration because last token
        # is generated by model and does not have a corresponding kv cache
        # TODO (Rohan138): This is inefficient; find a better way!
        prev_inputs["input_ids"] = inputs["input_ids"][:, :-1]
        prev_inputs["attention_mask"] = inputs["attention_mask"][:, :-1]
        prev_inputs["past_key_values"] = inputs.get("past_key_values", None)
        draft_outputs = draft_model(**prev_inputs, use_cache=True)

        draft_inputs = {
            "input_ids": inputs["input_ids"][:, -1:],
            "attention_mask": inputs["attention_mask"],
            "past_key_values": draft_outputs.past_key_values,
        }

        # Build staged speculative decoding tree
        tree = DecodingTree(draft_inputs, draft_model, temperature, topk, num_tokens)

        # Print tree statistics; uncomment to debug
        # print(f"Number of nodes: {count_nodes(tree.root)}")
        # print(f"Number of paths: {count_paths(tree.root)}")

        # depth-first search to get all input_ids
        tree_input_ids = get_tree_input_ids(tree.root)

        # pad input_ids that end prematurely in an eos token to length `depth + 1`
        for idx in range(len(tree_input_ids)):
            pad_length = num_tokens + 1 - tree_input_ids[idx].shape[1]
            if pad_length > 0:
                tree_input_ids[idx] = F.pad(tree_input_ids[idx], (0, pad_length))
        input_ids = torch.cat(tree_input_ids, dim=0)

        mask = inputs["attention_mask"]
        attention_mask = torch.cat(
            [mask, mask.new_ones((mask.shape[0], num_tokens + 1))], dim=-1
        )

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask[:, :-1].expand(
            input_ids.shape[0], -1
        )
        expanded_past_key_values = []
        for layer in model_outputs.past_key_values:
            expanded_layer = []
            for item in layer:
                expanded_item = item.expand(input_ids.shape[0], -1, -1, -1)
                expanded_layer.append(expanded_item)
            expanded_past_key_values.append(expanded_layer)
        model_inputs["past_key_values"] = expanded_past_key_values

        model_outputs = model(**model_inputs, use_cache=True)
        new_logits = model_outputs.logits

        if temperature is not None:
            new_logits = new_logits / temperature
            dist = torch.distributions.Categorical(logits=new_logits)
            selected_tokens = dist.sample()
        else:
            selected_tokens = new_logits.argmax(dim=-1)

        # find valid indices where input token predicted by
        # draft model is equal to previous output of model
        check_all = input_ids[:, 1:] == selected_tokens[:, :-1]
        valid_indices = torch.cumsum(check_all, dim=1)[:, -1]

        # choose the best valid index and find number of tokens generated
        chosen_index = valid_indices.argmax().item()
        num_generated = valid_indices[chosen_index].item() + 1
        chosen_tokens = selected_tokens[None, chosen_index, :num_generated]

        # update output sequence
        output_sequence = torch.cat([output_sequence, chosen_tokens], dim=1)
        last_token = chosen_tokens[:, -1:]
        total_length += num_generated

        if last_token == eos_token_id or total_length >= INP_LENGTH + GEN_LENGTH:
            break

        # update draft model inputs for next iteration
        last_input_token = input_ids[chosen_index, num_generated - 1]
        last_input_token = last_input_token.reshape(1, 1)
        inputs["input_ids"] = torch.cat([last_input_token, last_token], dim=1)
        inputs["attention_mask"] = attention_mask[:, :total_length]

        # index tree traversal by [chosen_index, depth] to get chosen_past_key_values
        tree_past_key_values = get_tree_past_key_values(tree.root)
        chosen_past_key_values = tree_past_key_values[chosen_index][num_generated - 1]
        inputs["past_key_values"] = chosen_past_key_values

        # update model past_key_values for next iteration
        chosen_model_past_key_values = []
        for layer in model_outputs.past_key_values:
            chosen_layer = []
            for item in layer:
                chosen_item = item[None, chosen_index, :, : total_length - 1]
                chosen_layer.append(chosen_item)
            chosen_model_past_key_values.append(chosen_layer)
        model_outputs.past_key_values = chosen_model_past_key_values

        # explicitly free up unused memory before next iteration
        torch.cuda.empty_cache()

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
