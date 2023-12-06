import time

import torch
from tqdm import tqdm

INP_LENGTH = 256
GEN_LENGTH = 128
NUM_TOKENS = 5
DEPTH = 2


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
        self.children = []
        self.build()

    def build(self):
        eos_token_id = self.draft_model.generation_config.eos_token_id

        outputs = self.draft_model(**self.inputs, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        if self.temperature is not None:
            next_token_logits = next_token_logits / self.temperature

        next_token_logits, next_token_ids = next_token_logits.topk(
            self.num_tokens, dim=-1
        )
        self.input_ids = next_token_ids
        self.is_eos_token = next_token_ids == eos_token_id
        self.attention_mask = torch.cat(
            (
                self.inputs["attention_mask"].expand(1, self.num_tokens, -1),
                (~self.is_eos_token).unsqueeze(-1).long(),
            ),
            dim=-1,
        )
        self.past_key_values = outputs.past_key_values
        self.next_token_logits = next_token_logits

    def get_next_inputs(self):
        return [
            {
                "input_ids": self.input_ids[:, idx],
                "attention_mask": self.attention_mask[:, idx],
                "past_key_values": self.past_key_values,
            }
            for idx in range(self.num_tokens)
        ]


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
        if depth > 1:
            for idx, next_input in enumerate(root.get_next_inputs()):
                if root.is_eos_token[:, idx].any():
                    continue
                decoding_tree = DecodingTree(
                    next_input,
                    self.draft_model,
                    self.temperature,
                    self.num_tokens,
                    depth - 1,
                )
                root.children.append(decoding_tree.root)
        return root


def count_nodes(root):
    return 1 + sum(count_nodes(child) for child in root.children)


def count_paths(root):
    if not root.children:
        return 1
    return sum(count_paths(child) for child in root.children)


def get_logits(root):
    if not root.children:
        return []
    logits = [root.next_token_logits.squeeze(0).tolist()]
    for child in root.children:
        logits.extend(get_logits(child))
    return logits

def get_paths(root: DecodingNode):
    if not root.children:
        return [root.get_next_inputs()]
    paths = []
    for child in root.children:
        paths.extend(get_paths(child))
    return paths

def get_paths_and_cum_logits(root: DecodingNode, accumulator=0.0):
    if not root.children:
        # Root is a terminal node
        # Return the transformer inputs for this node
        # Return the accumulated logit prob for this node
        return [root.get_next_inputs()], [accumulator]

    child_logits = root.next_token_logits.squeeze(0).tolist()
    paths = []
    cum_logits = []
    for child, child_logit in zip(root.children, child_logits):
        child_paths, child_cum_logits = get_paths_and_cum_logits(
            child, accumulator + child_logit
        )
        paths.extend(child_paths)
        cum_logits.extend(child_cum_logits)
    return paths, cum_logits


def staged_speculative_decoding(
    inputs, model, draft_model, temperature=None, num_tokens=NUM_TOKENS, depth=DEPTH
):
    # Generate initial kv cache for model
    outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values

    while True:
        # Build staged speculative decoding tree
        decoding_tree = DecodingTree(
            inputs, draft_model, temperature, num_tokens, depth
        )
        # Print tree statistics; uncomment to debug
        print(f"Number of nodes: {count_nodes(decoding_tree.root)}")
        print(f"Number of paths: {count_paths(decoding_tree.root)}")
        print(f"Node logits: {get_logits(decoding_tree.root)}")

        # depth-first search to get all paths ending in terminal nodes
        paths = get_paths(decoding_tree.root)
        # pad all paths to have length `num_tokens`
        for path in paths:
            # Note: this should be true iff there is eos at end of path
            if len(path) < num_tokens:
                path = path + [path[-1]] * (num_tokens - len(path))

        # Concatenate List[List[Dict[Tensor]]] into Dict[Tensor]
        input_ids = torch.stack(
            [torch.stack([x["input_ids"] for x in path]) for path in paths]
        )
        attention_mask = torch.stack(
            [torch.stack([x["attention_mask"] for x in path]) for path in paths]
        )
        # fold all but last dim into batch dimension
        B, T, _ = input_ids.shape
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        # Expand past_key_values to match input_ids
        expanded_past_key_values = []
        for layer in past_key_values:
            expanded_layer = []
            for item in layer:
                expanded_item = item.expand(input_ids.shape[0], -1, -1, -1)
                expanded_layer.append(expanded_item)
            expanded_past_key_values.append(expanded_layer)
        past_key_values = expanded_past_key_values

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        model_outputs = model(**model_inputs, use_cache=True)

        new_logits = model_outputs.logits[:, -num_tokens - 1 :]
        if temperature is not None:
            new_logits = new_logits / temperature
            dist = torch.distributions.Categorical(logits=new_logits)
            selected_tokens = dist.sample()
        else:
            selected_tokens = new_logits.argmax(dim=-1)
        
        candidate_tokens = input_ids.reshape(B, T, -1)
        breakpoint()


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
