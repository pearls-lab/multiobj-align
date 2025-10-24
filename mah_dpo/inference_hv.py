"""
Inference script for running multi-head DPO models on UltraFeedback dataset.
Supports base model, SFT model, and DPO model with specialized heads or ensemble generation.
"""

import argparse
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Dict, Optional, Union, Tuple
import os
import re
import json
import random
import tqdm
import time
import logging
import numpy as np
from multihead_model import MultiHeadCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import tempfile
from pathlib import Path


# Choose a generation seed when none is provided, so that each run differs by default
def _auto_generation_seed(cli_seed: Optional[int]) -> int:
    if cli_seed is not None:
        return cli_seed
    # mirror the idea from the ORM driver that picks a time based seed
    return int(time.time()) ^ os.getpid()


def set_generation_seed(generation_seed: int):
    """Set random seeds for model generation (not data selection)."""
    torch.manual_seed(generation_seed)
    torch.cuda.manual_seed(generation_seed)
    torch.cuda.manual_seed_all(generation_seed)
    np.random.seed(generation_seed)
    random.seed(generation_seed)

    # Important: Don't set deterministic=True for generation randomness
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def download_from_hf_hub(
    hf_path: str, filename: str = "policy.pt", local_cache_dir: Optional[str] = None
) -> str:
    """Download a file from Hugging Face Hub to a local cache directory."""
    from huggingface_hub import hf_hub_download

    if hf_path.startswith("hf://"):
        path_parts = hf_path[5:].split("/")
        if len(path_parts) >= 3:
            repo_id = "/".join(path_parts[:2])
            filename = "/".join(path_parts[2:])
        else:
            repo_id = "/".join(path_parts)
    else:
        repo_id = hf_path

    logger.info(f"Downloading from Hugging Face Hub: {repo_id}/{filename}")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=local_cache_dir
        )
        logger.info(f"Successfully downloaded from HF Hub to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Error downloading file from HF Hub: {e}")
        raise


def is_hf_hub_path(path: str) -> bool:
    """Check if a path is a Hugging Face Hub path."""
    if path.startswith("hf://"):
        return True

    if "/" in path and not path.startswith("/") and not path.startswith("./"):
        parts = path.split("/")
        if len(parts) == 2 and not any(
            part.endswith((".pt", ".pth", ".bin")) for part in parts
        ):
            return True
        if len(parts) >= 2 and not os.path.exists(path):
            return True

    return False


def extract_assistant_response(full_text):
    """Extract only the assistant's response from the full generated text."""
    # First try to remove any system prompt
    if "system\n" in full_text:
        full_text = full_text.split("system\n", 1)[1]

    # Try to find content after the last "Assistant:" or "assistant" tag
    assistant_pattern = re.compile(
        r"(?:^|\n)(?:Assistant:|assistant\n)(.*?)(?:$|\n\n(?:Human:|Human\n|user\n|$))",
        re.DOTALL | re.IGNORECASE,
    )
    matches = assistant_pattern.findall(full_text)

    if matches:
        # Return the last match (most recent assistant response)
        return matches[-1].strip()

    # If we didn't find a clear assistant marker, try splitting on 'user'
    if "user\n" in full_text:
        parts = full_text.split("user\n", 1)[1]
        if "assistant\n" in parts:
            return parts.split("assistant\n", 1)[1].strip()

    # Last resort - just return the whole text
    return full_text.strip()


def get_sample_queries_with_splits(
    total_samples: int = 500, total_splits: int = 5, split_id: int = 0, seed: int = 42
) -> List[Dict]:
    """
    Sample queries from UltraFeedback dataset and return the specified split.

    Args:
        total_samples: Total number of samples to draw from dataset
        total_splits: Total number of splits to divide samples into
        split_id: Which split to return (0-indexed)
        seed: Random seed for reproducibility

    Returns:
        List of query dictionaries for the specified split
    """
    # Set random seed for reproducibility
    random.seed(seed)

    logger.info("Loading UltraFeedback dataset...")
    # Load the dataset
    dataset = load_dataset("openbmb/UltraFeedback", split="train")

    total_available = len(dataset)
    logger.info(f"Found {total_available} samples in UltraFeedback dataset")

    # Sample the total number of samples with fixed seed
    if total_samples > total_available:
        logger.warning(
            f"Requested {total_samples} samples but only {total_available} available"
        )
        total_samples = total_available

    # Create indices for sampling
    all_indices = list(range(total_available))
    sampled_indices = random.sample(all_indices, total_samples)

    # Extract the sampled data
    sampled_data = []
    for idx in sampled_indices:
        sample = dataset[idx]
        sampled_data.append(
            {
                "instruction": sample["instruction"],
                "models": sample.get("models", []),
                "completions": sample.get("completions", []),
                "correct_answers": sample.get("correct_answers", []),
                "incorrect_answers": sample.get("incorrect_answers", []),
                "index": idx,
            }
        )

    # Calculate split size
    split_size = total_samples // total_splits
    remainder = total_samples % total_splits

    # Calculate start and end indices for this split
    start_idx = split_id * split_size
    if split_id < remainder:
        # First 'remainder' splits get one extra sample
        start_idx += split_id
        end_idx = start_idx + split_size + 1
    else:
        # Later splits use standard size
        start_idx += remainder
        end_idx = start_idx + split_size

    # Extract the split
    split_data = sampled_data[start_idx:end_idx]

    logger.info(
        f"📊 Split {split_id}/{total_splits-1}: {len(split_data)} queries (indices {start_idx}-{end_idx-1})"
    )

    return split_data


def load_ultrafeedback_dataset(
    num_samples: Optional[int] = None,
    seed: int = 42,
    split_index: Optional[int] = None,
    total_splits: int = 5,
):
    """
    Load the UltraFeedback dataset with optional subsampling and split processing.

    Args:
        num_samples: Number of samples to load. If None, load 500 samples.
        seed: Random seed for reproducible sampling.
        split_index: Which split to process (0 to total_splits-1). If None, process all data.
        total_splits: Total number of splits to divide the data into.

    Returns:
        List of examples with instruction and metadata.
    """
    if num_samples is None:
        num_samples = 500

    if split_index is not None:
        if split_index < 0 or split_index >= total_splits:
            raise ValueError(f"split_index must be between 0 and {total_splits-1}")

        # Get the specific split
        samples = get_sample_queries_with_splits(
            total_samples=num_samples,
            total_splits=total_splits,
            split_id=split_index,
            seed=seed,
        )
        logger.info(
            f"Processing split {split_index}/{total_splits-1} with {len(samples)} samples"
        )
    else:
        # Get all samples without splitting
        samples = get_sample_queries_with_splits(
            total_samples=num_samples, total_splits=1, split_id=0, seed=seed
        )
        logger.info(f"Processing all {len(samples)} samples")

    return samples


def run_inference_on_sample(
    model_type: str,
    model: Union[transformers.PreTrainedModel, MultiHeadCausalLM],
    tokenizer: transformers.PreTrainedTokenizer,
    instruction: str,
    max_length: Optional[int] = None,
    max_new_tokens: int = 1536,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = -1,
    head_index: Optional[int] = None,
    use_ensemble: bool = False,
    num_heads: int = 3,
    generation_seed: Optional[int] = None,
    sample_index: int = 0,
    head_weights: Optional[
        List[float]
    ] = None,) -> Dict[str, str]:
    """
    Run inference on a single UltraFeedback sample with controlled randomness.
    """
    # Set generation-specific seed if provided
    if generation_seed is not None:
        # Use different seed for each sample within the same generation
        sample_seed = generation_seed + sample_index
        set_generation_seed(sample_seed)
        if sample_index < 3:  # Only log for first few samples to avoid spam
            logger.info(
                f"🎯 Sample {sample_index}: Using seed {sample_seed} (base: {generation_seed})"
            )

    # Format with chat template - no system prompt
    chat_messages = [
        {"role": "user", "content": instruction},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)

    # Get device from model parameters instead of using model.device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generation parameters - ensure sampling is enabled
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,  # Controls how many new tokens to generate
        "do_sample": True,  # Critical: must be True for randomness
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    # Add top_k only if it's a positive integer (transformers doesn't accept -1 or 0)
    if top_k is not None and top_k > 0:
        generation_kwargs["top_k"] = top_k

    # Only add max_length if specified (when max_new_tokens alone isn't sufficient)
    if max_length is not None:
        generation_kwargs["max_length"] = max_length

    responses = {}

    try:
        with torch.no_grad():
            # Handle different model types and configurations
            if model_type in ["base", "sft"]:
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs,
                )

                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = extract_assistant_response(text)

                key = "base_model" if model_type == "base" else "sft_model"
                responses[key] = response

            elif model_type == "dpo":
                # If it's a DPO model with multiple heads, handle the different modes
                if use_ensemble:
                    # Generate with ensemble
                    try:
                        outputs = model.generate_ensemble(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            head_weights=head_weights, 
                            **generation_kwargs,
                        )

                        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = extract_assistant_response(text)
                        responses["ensemble"] = response
                    except Exception as e:
                        logger.error(f"Error in ensemble generation: {e}")
                        # Fall back to head 0
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            **generation_kwargs,
                        )

                        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = extract_assistant_response(text)
                        responses["head_0_fallback"] = response

                elif head_index is not None:
                    # Generate with specific head
                    try:
                        outputs = model.generate_with_head(
                            head_index,
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            **generation_kwargs,
                        )

                        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = extract_assistant_response(text)
                        responses[f"head_{head_index}"] = response
                    except Exception as e:
                        logger.error(f"Error generating with head {head_index}: {e}")
                        # Fall back to head 0
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            **generation_kwargs,
                        )

                        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = extract_assistant_response(text)
                        responses["head_0_fallback"] = response

                else:
                    # Generate with all heads
                    for h in range(num_heads):
                        try:
                            outputs = model.generate_with_head(
                                h,
                                inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                **generation_kwargs,
                            )

                            text = tokenizer.decode(
                                outputs[0], skip_special_tokens=True
                            )
                            response = extract_assistant_response(text)
                            responses[f"head_{h}"] = response
                        except Exception as e:
                            logger.error(f"Error generating with head {h}: {e}")
                            # Use head 0 as fallback if needed
                            if f"head_{h}" not in responses:
                                outputs = model.generate(
                                    inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    **generation_kwargs,
                                )

                                text = tokenizer.decode(
                                    outputs[0], skip_special_tokens=True
                                )
                                response = extract_assistant_response(text)
                                responses[f"head_{h}_fallback"] = response

    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        responses["error"] = f"Error: {str(e)}"

    return responses


def load_model(
    model_type: str,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    num_heads: int = 3,
    dtype: str = "bfloat16",
    cache_dir: Optional[str] = None,
) -> Tuple[
    Union[transformers.PreTrainedModel, MultiHeadCausalLM],
    transformers.PreTrainedTokenizer,
]:
    """
    Load the appropriate model based on the model type.

    Args:
        model_type: Type of model ('base', 'sft', or 'dpo')
        model_name: Name or path of the base model
        checkpoint_path: Path to checkpoint file (for 'sft' or 'dpo'), can be HF Hub path or local path
        num_heads: Number of heads for DPO model
        dtype: Model precision ('float16', 'float32', or 'bfloat16')
        cache_dir: Directory to cache downloaded files

    Returns:
        Tuple of (model, tokenizer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_type in ["base", "sft"]:
        logger.info(f"Loading {model_type} model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        # Load checkpoint for SFT model
        if model_type == "sft" and checkpoint_path:
            local_checkpoint_path = None

            if is_hf_hub_path(checkpoint_path):
                # Download from HF Hub if needed
                local_checkpoint_path = download_from_hf_hub(
                    checkpoint_path, filename="policy.pt", local_cache_dir=cache_dir
                )
            else:
                # Local path
                local_checkpoint_path = checkpoint_path

            logger.info(f"Loading SFT checkpoint: {local_checkpoint_path}")
            checkpoint = torch.load(local_checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if "state" in checkpoint:
                state_dict = checkpoint["state"]
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

    elif model_type == "dpo":
        logger.info(f"Loading base model for DPO: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # Create MultiHeadCausalLM model
        logger.info(f"Creating multi-head model with {num_heads} heads")
        model = MultiHeadCausalLM(base_model, num_heads=num_heads).to(device)

        # Load DPO checkpoint
        if checkpoint_path:
            local_checkpoint_path = None

            if is_hf_hub_path(checkpoint_path):
                # Download from HF Hub if needed
                local_checkpoint_path = download_from_hf_hub(
                    checkpoint_path, filename="policy.pt", local_cache_dir=cache_dir
                )
            else:
                # Local path - check if it's a directory
                if os.path.isdir(checkpoint_path):
                    local_checkpoint_path = os.path.join(checkpoint_path, "policy.pt")
                else:
                    local_checkpoint_path = checkpoint_path

            logger.info(f"Loading DPO checkpoint: {local_checkpoint_path}")
            checkpoint = torch.load(local_checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if "state" in checkpoint:
                state_dict = checkpoint["state"]
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()  # Set model to evaluation mode
    return model, tokenizer


def run_ultrafeedback_inference(
    model_type: str,
    model_name: str,
    output_file: str,
    checkpoint_path: Optional[str] = None,
    num_heads: int = 3,
    max_length: Optional[int] = None,
    max_new_tokens: int = 1536,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = -1,
    head_index: Optional[int] = None,
    use_ensemble: bool = False,
    num_samples: Optional[int] = None,
    dtype: str = "bfloat16",
    batch_size: int = 1,
    disable_tqdm: bool = False,
    cache_dir: Optional[str] = None,
    split_index: Optional[int] = None,
    total_splits: int = 5,
    seed: int = 42,
    generation_seed: Optional[int] = None,
    head_weights: Optional[
        List[float]
    ] = None,):
    """
    Run inference on UltraFeedback dataset and save results to a JSONL file.

    Args:
        model_type: Type of model ('base', 'sft', or 'dpo')
        model_name: Name or path of the base model
        output_file: Path to output JSONL file
        checkpoint_path: Path to checkpoint file (for 'sft' or 'dpo') - supports HF Hub or local paths
        num_heads: Number of heads in the DPO model
        max_length: Maximum context length
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter (0-1)
        top_k: Top-k sampling parameter
        head_index: For DPO model, which head to use (None to try all heads)
        use_ensemble: For DPO model, whether to use ensemble generation
        num_samples: Number of samples to use (None for 500)
        dtype: Model precision ('float16', 'float32', or 'bfloat16')
        batch_size: Batch size for processing
        disable_tqdm: Whether to disable the progress bar
        cache_dir: Directory to cache downloaded files
        split_index: Which split to process (0 to total_splits-1)
        total_splits: Total number of splits
        seed: Random seed for data sampling
        generation_seed: Random seed for model generation (different from data seed)
        head_weights: Custom weights for ensemble heads (None for equal weights)    
    """
    # Load the dataset with split handling
    examples = load_ultrafeedback_dataset(
        num_samples=num_samples,
        split_index=split_index,
        total_splits=total_splits,
        seed=seed,
    )

    # Load model and tokenizer
    model, tokenizer = load_model(
        model_type=model_type,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        num_heads=num_heads,
        dtype=dtype,
        cache_dir=cache_dir,
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create or truncate the output file
    with open(output_file, "w") as f:
        pass

    start_time = time.time()

    # Log the seeds being used
    logger.info(f"Data selection seed: {seed}")
    logger.info(f"Generation seed: {generation_seed}")
    if generation_seed is not None:
        logger.info(
            f"🎲 Using generation seed: {generation_seed} for randomized outputs"
        )
    else:
        logger.warning("⚠️  No generation seed set - outputs may be deterministic!")

    # Process examples with progress bar
    for idx, example in enumerate(tqdm.tqdm(examples, disable=disable_tqdm)):
        instruction = example["instruction"]
        original_index = example.get("index", idx)

        # Run inference
        responses = run_inference_on_sample(
            model_type=model_type,
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            head_index=head_index,
            use_ensemble=use_ensemble,
            num_heads=num_heads,
            generation_seed=generation_seed,
            sample_index=idx,
            head_weights=head_weights,        
        )

        # Get the primary response based on configuration
        primary_response = None
        if model_type in ["base", "sft"]:
            key = "base_model" if model_type == "base" else "sft_model"
            primary_response = responses.get(key, "")
        elif model_type == "dpo":
            if use_ensemble:
                primary_response = responses.get(
                    "ensemble", responses.get("head_0_fallback", "")
                )
            elif head_index is not None:
                primary_response = responses.get(
                    f"head_{head_index}", responses.get("head_0_fallback", "")
                )
            else:
                # Default to head 0 if no specific choice made
                primary_response = responses.get("head_0", "")

        # Create output record
        print(f"Primary response for sample {idx}: {primary_response}...")
        output_record = {
            "id": f"ultrafeedback_sample_{original_index}",
            "instruction": instruction,
            "model_response": primary_response,
            "all_responses": responses,
            "metadata": {
                "model_type": model_type,
                "head_index": head_index,
                "use_ensemble": use_ensemble,
                "split_index": split_index,
                "sample_index": idx,
                "original_index": original_index,
                "data_seed": seed,
                "generation_seed": generation_seed,
            },
        }

        # Append to output file
        with open(output_file, "a") as f:
            f.write(json.dumps(output_record) + "\n")

        # Print progress and sample responses every 10 samples
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(examples) - idx - 1)

            split_info = (
                f" (Split {split_index}/{total_splits-1})"
                if split_index is not None
                else ""
            )
            logger.info(f"Processed {idx + 1}/{len(examples)} examples{split_info}")
            logger.info(
                f"Avg time per example: {avg_time:.2f}s, Est. time remaining: {remaining:.2f}s"
            )

            if idx == 0:
                # Print the first sample's details
                logger.info("\n----- Sample Instruction -----")
                logger.info(
                    f"Instruction: {instruction[:200]}..."
                    if len(instruction) > 200
                    else instruction
                )
                logger.info("----- Sample Responses -----")
                for key, response in responses.items():
                    logger.info(
                        f"{key}: {response[:200]}..."
                        if len(response) > 200
                        else response
                    )
                logger.info("---------------------------\n")

    total_time = time.time() - start_time
    split_info = (
        f" for split {split_index}/{total_splits-1}" if split_index is not None else ""
    )
    logger.info(
        f"Completed inference on {len(examples)} examples{split_info} in {total_time:.2f}s"
    )
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on UltraFeedback dataset"
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "sft", "dpo"],
        required=True,
        help="Type of model to use",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name or path of the base model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model checkpoint (supports HF Hub: user/repo, or local path)",
    )
    parser.add_argument(
        "--num_heads", type=int, default=3, help="Number of heads in the DPO model"
    )
    parser.add_argument(
        "--cache_dir", type=str, help="Directory to cache downloaded files"
    )

    # Generation parameters
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum token limit (context + generation). If not specified, only max_new_tokens will be used.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1536,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (positive integer, -1 or 0 disables top-k)",
    )

    # Split processing parameters
    parser.add_argument(
        "--split_index",
        type=int,
        help="Which split to process (0 to total_splits-1). If not specified, process all data.",
    )
    parser.add_argument(
        "--total_splits",
        type=int,
        default=5,
        help="Total number of splits to divide the data into (default: 5)",
    )

    # Head selection
    parser.add_argument(
        "--head_index",
        type=int,
        help="Specific head to use for DPO model (0 to num_heads-1)",
    )
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        help="Use ensemble of all heads for DPO model",
    )
    parser.add_argument(
        "--head_weights",
        type=float,
        nargs="+",
        help="Custom weights for ensemble heads (e.g., --head_weights 0.5 0.5 0). Must match num_heads. If not specified, equal weights are used.",
    )

    # Seed parameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data sampling"
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        help="Random seed for model generation (different from data selection seed)",
    )

    # Other parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to use (default: 500)",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output file path (JSONL format)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Precision to use for model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--disable_tqdm", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()
    # Decide the run level generation seed
    effective_gen_seed = _auto_generation_seed(args.generation_seed)

    # Validate split parameters
    if args.split_index is not None:
        if args.split_index < 0 or args.split_index >= args.total_splits:
            raise ValueError(f"split_index must be between 0 and {args.total_splits-1}")

    # Validate head_weights parameter
    if args.head_weights is not None:
        if len(args.head_weights) != args.num_heads:
            raise ValueError(
                f"Number of head weights ({len(args.head_weights)}) must match num_heads ({args.num_heads})"
            )
        if not args.use_ensemble:
            raise ValueError("--head_weights can only be used with --use_ensemble")

    # Run inference
    run_ultrafeedback_inference(
        model_type=args.model_type,
        model_name=args.model_name,
        output_file=args.output_file,
        checkpoint_path=args.checkpoint_path,
        num_heads=args.num_heads,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        head_index=args.head_index,
        use_ensemble=args.use_ensemble,
        num_samples=args.num_samples,
        dtype=args.dtype,
        batch_size=args.batch_size,
        disable_tqdm=args.disable_tqdm,
        cache_dir=args.cache_dir,
        split_index=args.split_index,
        total_splits=args.total_splits,
        seed=args.seed,
        generation_seed=effective_gen_seed, - Use effective generation seed
        head_weights=args.head_weights,    
    )


if __name__ == "__main__":
    main()
