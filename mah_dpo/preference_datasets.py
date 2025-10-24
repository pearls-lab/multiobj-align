########################
# This script is modified from DPO codebase https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py
########################
import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import os
import json


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, "html.parser")

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == "p":
            text.append(
                "".join(
                    child.string
                    for child in element.children
                    if isinstance(child, NavigableString)
                )
            )
        elif element.name == "pre":
            for code in element.find_all("code"):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == "code":
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(
    split, silent=False, cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.

    We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f"Loading SE dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset(
        "HuggingFaceH4/stack-exchange-preferences", cache_dir=cache_dir
    )["train"]
    print("done")

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = (
        dataset.select(range(int(len(dataset) * 0.01)))
        if split == "test"
        else dataset.select(range(int(len(dataset) * 0.01), len(dataset)))
    )

    def strip_html(x):
        x["question"] = strip_html_tags(x["question"])
        for a in x["answers"]:
            a["text"] = strip_html_tags(a["text"])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc="Processing SE", disable=silent):
        prompt = "\n\nHuman: " + row["question"] + "\n\nAssistant:"
        responses = [" " + a["text"] for a in row["answers"]]
        scores = [a["pm_score"] for a in row["answers"]]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]["responses"] = responses
        data[prompt]["pairs"] = pairs
        data[prompt]["sft_target"] = max(
            responses, key=lambda x: scores[responses.index(x)]
        )

    return data


def get_shp(
    split: str, silent: bool = False, cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

    We filter preference pairs to only keep pairs where the score ratio is at least 2.
    For this dataset, the sft_target is the response with the highest score.
    """
    print(f"Loading SHP dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset("stanfordnlp/SHP", split=split, cache_dir=cache_dir)
    print("done")

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing SHP", disable=silent):
        prompt = "\n\nHuman: " + row["history"] + "\n\nAssistant:"
        responses = [" " + row["human_ref_A"], " " + row["human_ref_B"]]
        scores = [row["score_A"], row["score_B"]]
        if prompt in data:
            n_responses = len(data[prompt]["responses"])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]["pairs"].append(
            (n_responses, n_responses + 1)
            if row["labels"] == 1
            else (n_responses + 1, n_responses)
        )
        data[prompt]["responses"].extend(responses)
        data[prompt]["scores"].extend(scores)

    for prompt in data:
        data[prompt]["sft_target"] = max(
            data[prompt]["responses"],
            key=lambda x: data[prompt]["scores"][data[prompt]["responses"].index(x)],
        )
        del data[prompt]["scores"]

    return data


def get_hh(
    split: str, silent: bool = False, cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt1': {
            'responses': List[str],
            'pairs': List[Tuple[int, int]],
            'sft_target': str
        },
        'prompt2': {
            ...
        },
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.

    For this dataset, the sft_target is just the chosen response.
    """
    print(f"Loading HH dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset(
        "Anthropic/hh-rlhf", split=split, cache_dir=cache_dir
    )
    print("done")

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex["chosen"])
        chosen_response = ex["chosen"][len(prompt) :]
        rejected_response = ex["rejected"][len(prompt) :]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing HH", disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    return data


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_math_dataset(
    filepath: str, tokenizer, split: str, silent: bool = False, max_samples: int = None
):
    """Load a custom dataset from a filepath and format with chat template.

    Args:
        filepath: Path to the JSON file
        tokenizer: Tokenizer to use for chat templates
        split: Which split to use (train/test)
        silent: Whether to silence progress bars
        max_samples: Maximum number of samples to use (randomly selected)
    """
    print(f"Loading custom dataset ({split} split) from {filepath}...")

    try:
        with open(filepath, "r") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from custom dataset")
    except Exception as e:
        raise Exception(f"Error loading custom dataset: {e}")

    # Create proper train/test split FIRST
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)  # 90% train, 10% test
    if split == "train":
        dataset = dataset[:split_idx]
        if max_samples and max_samples < len(dataset):
            dataset = random.sample(dataset, max_samples)
    else:  # test
        dataset = dataset[split_idx:]
        dataset = dataset[:256]  # Max 256 test examples

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing custom dataset", disable=silent):
        # Format the prompt using chat template
        chat_messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}",
            },
            {"role": "user", "content": row["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        # Get chosen and rejected responses
        chosen = row["chosen"]
        rejected = row["rejected"]

        # Add to dataset
        responses = [chosen, rejected]
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    return data


def get_sm_dataset(
    filepath: str, tokenizer, split: str, silent: bool = False, max_samples: int = None
):
    """Load a custom dataset from a filepath and format with chat template.

    Args:
        filepath: Path to the JSON file
        tokenizer: Tokenizer to use for chat templates
        split: Which split to use (train/test)
        silent: Whether to silence progress bars
        max_samples: Maximum number of samples to use (randomly selected)
    """
    print(f"Loading custom dataset ({split} split) from {filepath}...")

    try:
        with open(filepath, "r") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from custom dataset")
    except Exception as e:
        raise Exception(f"Error loading custom dataset: {e}")

    # Create proper train/test split FIRST
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)  # 90% train, 10% test
    if split == "train":
        dataset = dataset[:split_idx]
        if max_samples and max_samples < len(dataset):
            dataset = random.sample(dataset, max_samples)
    else:  # test
        dataset = dataset[split_idx:]
        dataset = dataset[:256]  # Max 256 test examples

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing custom dataset", disable=silent):
        # Format the prompt using chat template
        chat_messages = [
            {
                "role": "system",
                "content": "You are a tutor who is helping a beginner student learn programming. Continue as the same tutor and reply similarly to the last student message, matching EXACTLY the SAME speaking tone and tutoring style as in your earlier messages (e.g. reply to the student's last message concisely in 1-2 sentences and then always ask a meaningful follow-up question).",
            },
            {"role": "user", "content": row["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        # Get chosen and rejected responses
        chosen = row["chosen"]
        rejected = row["rejected"]

        # Add to dataset
        responses = [chosen, rejected]
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    return data


def get_ultra_dataset(
    filepath: str, tokenizer, split: str, silent: bool = False, max_samples: int = None
):
    """Load an ultra dataset from a filepath and format with chat template.

    Designed for ultra_help.json, ultra_hon.json, ultra_truth.json, and ultra_combined.json
    files. Uses a simpler chat template without system message.

    Args:
        filepath: Path to the JSON file
        tokenizer: Tokenizer to use for chat templates
        split: Which split to use (train/test)
        silent: Whether to silence progress bars
        max_samples: Maximum number of samples to use (randomly selected)
    """
    print(f"Loading ultra dataset ({split} split) from {filepath}...")

    try:
        with open(filepath, "r") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from ultra dataset")
    except Exception as e:
        raise Exception(f"Error loading ultra dataset: {e}")

    # Create proper train/test split FIRST
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)  # 90% train, 10% test
    if split == "train":
        dataset = dataset[:split_idx]
        if max_samples and max_samples < len(dataset):
            dataset = random.sample(dataset, max_samples)
    else:  # test
        dataset = dataset[split_idx:]
        dataset = dataset[:256]  # Max 256 test examples

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing ultra dataset", disable=silent):
        # Format the prompt using simplified chat template (no system message)
        chat_messages = [
            {"role": "user", "content": row["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        # Get chosen and rejected responses
        chosen = row["chosen"]
        rejected = row["rejected"]

        # Add to dataset
        responses = [chosen, rejected]
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    return data


def get_dataset(
    name: str,
    tokenizer,
    split: str,
    silent: bool = False,
    cache_dir: str = None,
    model_name: Optional[str] = None,
    max_samples: int = None,
):
    """Load the given dataset by name."""
    if name.endswith(".json"):
        # Check if it's an ultra dataset
        if name in [
            "ultra_help.json",
            "ultra_hon.json",
            "ultra_truth.json",
            "ultra_combined_clean.json",
        ]:
            data = get_ultra_dataset(
                name, tokenizer, split, silent=silent, max_samples=max_samples
            )
        elif name in [
            "sm_acc.json",
            "sm_eng.json",
        ]:
            data = get_sm_dataset(
                name, tokenizer, split, silent=silent, max_samples=max_samples
            )
        else:
            data = get_math_dataset(
                name, tokenizer, split, silent=silent, max_samples=max_samples
            )
    elif name == "shp":
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == "hh":
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == "se":
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    # Verify dataset structure
    assert set(list(data.values())[0].keys()) == {
        "responses",
        "pairs",
        "sft_target",
    }, f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(
    tokenizer,
) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

    The collate function takes a list of examples (dicts, where values are lists of
      ints [tokens] or strings [the original texts]) and returns a batch of examples,
      PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if (
                    "prompt" in k
                ):  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if (
                    "prompt" in k
                ):  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn


def tokenize_batch_element(
    prompt: str,
    chosen: str,
    rejected: str,
    truncation_mode: str,
    tokenizer,
    max_length: int,
    max_prompt_length: int,
) -> Dict:
    """Tokenize a single batch element.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
      in case the prompt + chosen or prompt + rejected responses is/are too long. First
      we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
      the sum of the length of the prompt and the chosen/rejected response, with -100 for the
      prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    # assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    # assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)

    rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(
        len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
    )

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        if truncation_mode == "keep_start":
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == "keep_end":
            prompt_tokens = {
                k: v[-max_prompt_length:] for k, v in prompt_tokens.items()
            }
        else:
            raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        chosen_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()
        }
        rejected_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()
        }

    # Create labels
    chosen_sequence_tokens = {
        k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens
    }
    rejected_sequence_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
        -100
    ] * len(prompt_tokens["input_ids"])

    batch = {}

    batch["prompt"] = prompt
    batch["chosen"] = prompt + chosen
    batch["rejected"] = prompt + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    return batch


def get_batch_iterator(
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    sft_mode: bool = False,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    max_samples: int = 10000,
) -> Iterator[Dict]:
    """Get an iterator over balanced batches ensuring each dimension is equally represented."""
    assert (
        n_epochs is not None or n_examples is not None
    ), "Must specify either n_epochs or n_examples"
    print(f"Creating iterator with datasets in this order: {names}")
    dimension_map = {name: idx for idx, name in enumerate(names)}
    print(f"Dimension mapping: {dimension_map}")

    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    # Load datasets and organize by dimension
    dimension_data = []
    for dim_idx, name in enumerate(names):
        truncation_mode = "keep_end" if name == "hh" else "keep_start"
        dataset = get_dataset(
            name,
            tokenizer,
            split,
            silent=silent,
            cache_dir=cache_dir,
            model_name=model_name,
            max_samples=max_samples,
        )

        # Convert to flat list of examples with dimension index
        dim_data = []
        for prompt, data in dataset.items():
            for pair in data["pairs"]:
                dim_data.append(
                    {
                        "prompt": prompt,
                        "chosen_idx": pair[0],
                        "rejected_idx": pair[1],
                        "responses": data["responses"],
                        "sft_target": data["sft_target"],
                        "truncation_mode": truncation_mode,
                        "dimension": dim_idx,
                    }
                )

        dimension_data.append(dim_data)
        if not silent:
            print(f"Loaded {len(dim_data)} examples for dimension {dim_idx} ({name})")

    # Balance datasets
    min_size = min(len(data) for data in dimension_data)
    for dim_idx, data in enumerate(dimension_data):
        if len(data) > min_size:
            with TemporarilySeededRandom(seed + dim_idx):
                random.shuffle(data)
                dimension_data[dim_idx] = data[:min_size]

    if not silent:
        print(f"Balanced to {min_size} examples per dimension")

    # Return batches
    collate_fn = get_collate_fn(tokenizer)
    example_idx = 0
    epoch_idx = 0
    num_heads = len(names)

    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            break

        # Shuffle each dimension independently
        for dim_idx, dim_data in enumerate(dimension_data):
            with TemporarilySeededRandom(seed + epoch_idx + dim_idx):
                random.shuffle(dimension_data[dim_idx])

        # Create mini-batches with balanced examples from each dimension
        # This ensures each batch contains examples from all dimensions
        batched_data = []
        batch_size_per_dim = max(1, batch_size // num_heads)

        # Create batches with internal balance
        for batch_start in range(0, min_size, batch_size_per_dim):
            batch_end = min(batch_start + batch_size_per_dim, min_size)
            batch = []

            # Add examples from each dimension
            for dim_idx in range(num_heads):
                batch.extend(dimension_data[dim_idx][batch_start:batch_end])

            # Shuffle the batch to avoid having all examples from one dimension together
            with TemporarilySeededRandom(seed + epoch_idx + batch_start):
                random.shuffle(batch)

            # Add the balanced batch if it's not empty
            if batch:
                batched_data.append(batch)

        # Flatten the batches for processing while preserving balance
        interleaved_data = []
        for batch in batched_data:
            interleaved_data.extend(batch)

        if not silent:
            print(
                f"Created {len(interleaved_data)} examples with balanced dimensions in {len(batched_data)} batches"
            )
            # Count examples per dimension
            dim_counts = defaultdict(int)
            for item in interleaved_data:
                dim_counts[item["dimension"]] += 1
            for dim_idx in range(num_heads):
                print(
                    f"Dimension {dim_idx} ({names[dim_idx]}) has {dim_counts[dim_idx]} examples in epoch"
                )

        # Create batches
        for i in range(0, len(interleaved_data), batch_size):
            if n_examples is not None and example_idx >= n_examples:
                if not silent:
                    print(f"Reached {n_examples} examples")
                return

            batch_data = interleaved_data[i : i + batch_size]
            batch = []
            dimensions = []

            for item in batch_data:
                if sft_mode:
                    batch_element = tokenize_batch_element(
                        item["prompt"],
                        item["sft_target"],
                        item["sft_target"],
                        item["truncation_mode"],
                        tokenizer,
                        max_length,
                        max_prompt_length,
                    )
                    batch_element = {
                        k: v for k, v in batch_element.items() if "rejected" not in k
                    }
                else:
                    batch_element = tokenize_batch_element(
                        item["prompt"],
                        item["responses"][item["chosen_idx"]],
                        item["responses"][item["rejected_idx"]],
                        item["truncation_mode"],
                        tokenizer,
                        max_length,
                        max_prompt_length,
                    )

                batch.append(batch_element)
                dimensions.append(item["dimension"])
                example_idx += 1

            if batch:
                # Verify balance in this batch
                dim_counts = defaultdict(int)
                for dim in dimensions:
                    dim_counts[dim] += 1

                # Log batch balance but only occasionally to avoid spam
                if i % (10 * batch_size) == 0 and not silent:
                    balance_info = ", ".join(
                        [f"{names[dim]}: {count}" for dim, count in dim_counts.items()]
                    )
                    print(f"Batch dimension balance: {balance_info}")

                collated = collate_fn(batch)
                collated["dimensions"] = dimensions
                yield collated

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != " " and str_b[idx] != " ":
                return False
            else:
                if str_a[idx] == " ":
                    str_a = str_a[:idx] + str_a[idx + 1 :]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1 :]

    return True
