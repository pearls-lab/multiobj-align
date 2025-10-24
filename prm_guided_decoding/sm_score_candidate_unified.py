#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import logging

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class UnifiedRewardModel:
    """
    Wrapper for unified reward model that provides step-level scoring
    Compatible with the existing value guidance interface

    This is designed for binary classification reward models (2 labels).
    Returns the probability of the positive class (label 1) as the reward score.
    """

    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.step_sep_token_id = None

    def to(self, device):
        """Move model to device"""
        self.base_model = self.base_model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode"""
        self.base_model.eval()
        return self

    def parameters(self):
        """Return model parameters"""
        return self.base_model.parameters()

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """Load unified reward model from pretrained"""
        # Extract tokenizer and base_model_name if provided in kwargs
        tokenizer = kwargs.pop("tokenizer", None)
        base_model_name = kwargs.pop("base_model_name", None)

        # Load the sequence classification model (binary classification)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            problem_type="single_label_classification",
            **kwargs,
        )

        # Load tokenizer if not provided
        if tokenizer is None:
            # For unified reward models, prioritize the base model used during training
            # Since the saved unified model doesn't have a tokenizer, load from meta-llama/Llama-3.1-8B
            tokenizer_sources = []

            # First try the base model used for training unified models
            tokenizer_sources.append("meta-llama/Llama-3.1-8B")

            # Then try the provided base_model_name if different
            if base_model_name and base_model_name != "meta-llama/Llama-3.1-8B":
                tokenizer_sources.append(base_model_name)

            # Try the model path itself (unlikely to work for unified models but worth trying)
            tokenizer_sources.append(model_name_or_path)

            # Add common fallbacks
            tokenizer_sources.extend(
                [
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct",
                ]
            )

            tokenizer = None
            for source in tokenizer_sources:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        source, trust_remote_code=True
                    )
                    logger.info(f"Successfully loaded tokenizer from: {source}")
                    break
                except Exception as e:
                    logger.warning(
                        f"WARNING: Could not load tokenizer from {source}: {e}"
                    )
                    continue

            if tokenizer is None:
                raise ValueError("Could not load tokenizer from any source")

        # Set up tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Add default chat template if missing
        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n\n{% endif %}{% endfor %}"
            logger.info("Added default chat template to tokenizer")

        model = cls(base_model, tokenizer)

        # Set step separator token ID if available
        step_sep_token = "<extra_0>"
        try:
            model.step_sep_token_id = tokenizer.encode(step_sep_token)[0]
            logger.info(f"Set step separator token ID: {model.step_sep_token_id}")
        except:
            logger.warning(
                f"WARNING: Could not set step separator token ID for {step_sep_token}"
            )
            model.step_sep_token_id = None

        return model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through the unified model (binary classification)"""
        # Get model device
        model_device = next(self.base_model.parameters()).device

        # Move inputs to model device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if attention_mask is not None and attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)

        # Forward pass
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        # For binary classification, extract probability of positive class (class 1)
        logits = outputs.logits  # Shape: [batch_size, 2]
        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
        scores = probs[
            :, 1:2
        ]  # Extract probability of class 1, keep dimension [batch_size, 1]

        return type(
            "UnifiedOutput",
            (),
            {
                "logits": scores,  # Return positive class probability as logits for compatibility
                "last_hidden_state": getattr(outputs, "hidden_states", None),
            },
        )()

    def score_conversation(self, query: str, response: str) -> float:
        """
        Score a complete conversation using the unified reward model.
        Returns the probability of positive class (good response) as a float in [0, 1].
        """
        # Build conversation exactly like training data format
        conversation = f"User: {query}\n\nAssistant: {response}"

        input_ids = self.tokenizer.encode(conversation, return_tensors="pt").to(
            next(self.base_model.parameters()).device
        )
        attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        with torch.no_grad():
            # Get unified model prediction for conversation
            outputs = self.forward(input_ids, attention_mask)
            conversation_score = outputs.logits.squeeze().cpu().item()

        # Clean up tensors
        del input_ids, attention_mask
        torch.cuda.empty_cache()

        return conversation_score

    def score_multiple_conversations(
        self, query: str, responses: List[str]
    ) -> List[float]:
        """Score multiple conversations efficiently."""
        if not responses:
            return []

        scores = []
        for response in responses:
            score = self.score_conversation(query, response)
            scores.append(score)

        return scores


def format_conversation_with_candidate(
    messages: List[Dict[str, str]], candidate: str
) -> str:
    """
    Format a conversation for unified reward model scoring.
    Follows the User:/Assistant: format with proper spacing used in training.
    """
    # Build the conversation history first
    conversation_parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            conversation_parts.append(f"User: {content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    # Join with proper spacing and add the candidate response
    conversation = "\n\n".join(conversation_parts)

    # Add the candidate as the final assistant response
    if conversation:
        conversation += f"\n\nAssistant: {candidate}"
    else:
        conversation = f"Assistant: {candidate}"

    return conversation


def load_unified_model_and_tokenizer(model_name: str, device_id: int = None):
    """Load unified reward model on specific device."""
    logger.info(f"Loading unified reward model: {model_name}")

    try:
        # Load unified reward model
        if device_id is not None:
            device = f"cuda:{device_id}"
            model = UnifiedRewardModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
            ).eval()
        else:
            # Use device_map="auto" for model parallelism across GPUs
            model = UnifiedRewardModel.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()

        tokenizer = model.tokenizer
        logger.info(f"Successfully loaded unified reward model: {model_name}")
        return model, tokenizer

    except Exception as e:
        logger.error(f"ERROR: Failed to load unified reward model: {e}")
        raise


def score_conversations_unified(
    model, tokenizer, conversations: List[str], batch_size: int = 8
) -> List[float]:
    """Score conversations using the unified reward model."""
    scores: List[float] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(conversations), batch_size):
            batch_conversations = conversations[i : i + batch_size]

            # Encode conversations
            encoded = tokenizer(
                batch_conversations,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=2048,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            # Use autocast for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model.forward(encoded["input_ids"], encoded["attention_mask"])
                batch_scores = outputs.logits.squeeze().detach().float().cpu()

                # Handle single item case
                if batch_scores.dim() == 0:
                    batch_scores = [batch_scores.item()]
                else:
                    batch_scores = batch_scores.tolist()

                scores.extend(batch_scores)

            # Clean up
            del encoded, outputs
            torch.cuda.empty_cache()

    return scores


def extract_query_from_messages(messages: List[Dict[str, str]]) -> str:
    """Extract the main query/question from messages for unified scoring."""
    # Find the first user message as the main query
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")

    # Fallback: use the first assistant message if no user message
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")

    return ""


def process_worker_unified(args_tuple):
    """Worker function for processing a chunk of data on a specific GPU with unified model."""
    chunk_data, model_name, max_length, batch_size, device_id, worker_id = args_tuple

    # Initialize CUDA in this process
    torch.cuda.set_device(device_id)

    # Load unified model on this GPU
    model, tokenizer = load_unified_model_and_tokenizer(model_name, device_id=device_id)

    results = []

    for row_data in chunk_data:
        messages, candidates, row = row_data

        # Extract query for unified scoring
        query = extract_query_from_messages(messages)

        # Build conversation history (without candidates)
        conversation_history = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                conversation_history.append(f"User: {content}")
            elif role == "assistant":
                conversation_history.append(f"Assistant: {content}")

        base_conversation = "\n\n".join(conversation_history)

        # Score each candidate using unified model approach
        candidate_conversations = []
        for candidate in candidates:
            if base_conversation:
                full_conversation = f"{base_conversation}\n\nAssistant: {candidate}"
            else:
                full_conversation = f"Assistant: {candidate}"
            candidate_conversations.append(full_conversation)

        # Score all candidates
        scores = score_conversations_unified(
            model, tokenizer, candidate_conversations, batch_size=batch_size
        )

        # Find best candidate
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_score = float(scores[best_idx])
        best_candidate = candidates[best_idx]

        out_obj = {
            **row,
            "rm_model": model_name,
            "scores": scores,
            "best_index": best_idx,
            "best_score": best_score,
            "best_candidate": best_candidate,
            "query": query,
        }
        results.append(out_obj)

        logger.info(
            f"[Worker {worker_id}] Processed id={row.get('id', '')} best_idx={best_idx} best_score={best_score:.4f}"
        )

    return results


def load_input_data(input_path: Path, num_samples: int = None, seed: int = None):
    """Load input data, optionally sampling a subset."""
    data = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            messages = row.get("messages", [])
            candidates: List[str] = row.get("candidates", [])
            if not isinstance(candidates, list) or len(candidates) == 0:
                continue

            data.append((messages, candidates, row))

    # Sample if requested
    if num_samples is not None and len(data) > num_samples:
        if seed is not None:
            random.seed(seed)
        data = random.sample(data, num_samples)
        logger.info(f"Sampled {num_samples} rows from {len(data)} total (seed={seed})")

    return data


def chunk_data(data, num_chunks):
    """Split data into roughly equal chunks."""
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end

    return chunks


def process_file_parallel_unified(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_length: int = 2048,
    batch_size: int = 8,
    num_gpus: int = 8,
    num_samples: int = None,
    seed: int = None,
):
    """Process file using multiple GPUs in parallel with unified reward model."""
    logger.info(f"Loading input data from {input_path}")
    data = load_input_data(input_path, num_samples=num_samples, seed=seed)

    if not data:
        logger.error("No valid data found!")
        return

    logger.info(f"Processing {len(data)} rows")
    logger.info(f"Using {num_gpus} GPUs with batch size {batch_size}")
    logger.info(f"Using unified reward model: {model_name}")

    # Split data into chunks for each GPU
    chunks = chunk_data(data, num_gpus)

    # Prepare arguments for each worker
    worker_args = []
    for i, chunk in enumerate(chunks):
        if chunk:  # Only create worker if chunk has data
            worker_args.append((chunk, model_name, max_length, batch_size, i, i))

    # Open output file for real-time writing
    completed_count = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        # Process in parallel
        with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
            future_to_worker = {
                executor.submit(process_worker_unified, args): i
                for i, args in enumerate(worker_args)
            }

            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    results = future.result()

                    # Write results immediately as they complete
                    for result in results:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()  # Force write to disk
                        completed_count += 1

                        # Print progress for each completed item
                        logger.info(
                            f"[Completed {completed_count}/{len(data)}] Worker {worker_id} | "
                            f"best_idx={result['best_index']} best_score={result['best_score']:.4f} "
                            f"id={result.get('id', '')}"
                        )

                    logger.info(
                        f"Worker {worker_id} finished with {len(results)} results"
                    )

                except Exception as exc:
                    logger.error(f"Worker {worker_id} generated an exception: {exc}")

    logger.info(f"Completed! Scored {completed_count} rows → {output_path}")


def process_file_single_unified(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_length: int = 2048,
    batch_size: int = 16,
    num_samples: int = None,
    seed: int = None,
):
    """Alternative: Load one unified model across all GPUs and process with larger batches."""
    logger.info(f"Loading unified model {model_name} across all available GPUs")
    model, tokenizer = load_unified_model_and_tokenizer(
        model_name
    )  # Uses device_map="auto"

    logger.info(f"Loading input data from {input_path}")
    data = load_input_data(input_path, num_samples=num_samples, seed=seed)

    if not data:
        logger.error("No valid data found!")
        return

    logger.info(f"Processing {len(data)} rows with batch size {batch_size}")

    written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for i, (messages, candidates, row) in enumerate(data):
            # Extract query for logging/debugging
            query = extract_query_from_messages(messages)

            # Build conversation history
            conversation_history = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    conversation_history.append(f"User: {content}")
                elif role == "assistant":
                    conversation_history.append(f"Assistant: {content}")

            base_conversation = "\n\n".join(conversation_history)

            # Prepare candidate conversations
            candidate_conversations = []
            for candidate in candidates:
                if base_conversation:
                    full_conversation = f"{base_conversation}\n\nAssistant: {candidate}"
                else:
                    full_conversation = f"Assistant: {candidate}"
                candidate_conversations.append(full_conversation)

            # Score candidates
            scores = score_conversations_unified(
                model, tokenizer, candidate_conversations, batch_size=batch_size
            )

            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            best_score = float(scores[best_idx])
            best_candidate = candidates[best_idx]

            out_obj = {
                **row,
                "rm_model": model_name,
                "scores": scores,
                "best_index": best_idx,
                "best_score": best_score,
                "best_candidate": best_candidate,
                "query": query,
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()  # Force write to disk immediately
            written += 1

            logger.info(
                f"[Completed {written}/{len(data)}] best_idx={best_idx} best_score={best_score:.4f} id={row.get('id', '')}"
            )

    logger.info(f"Completed! Scored {written} rows → {output_path}")


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Score preference candidates with unified reward model using multiple GPUs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with messages and candidates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL with scores and best candidate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Jennny/unified_rm_1e5_1600",
        help="Hugging Face unified reward model repo",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of random samples to process (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--strategy",
        choices=["multi_process", "model_parallel"],
        default="multi_process",
        help="Strategy: multi_process (separate model per GPU) or model_parallel (one model across all GPUs)",
    )
    args = parser.parse_args()

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    available_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {available_gpus}")

    if args.num_gpus > available_gpus:
        logger.warning(
            f"Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}."
        )
        args.num_gpus = available_gpus

    logger.info(f"Using unified reward model: {args.model}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")

    if args.strategy == "multi_process":
        process_file_parallel_unified(
            args.input,
            args.output,
            args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        process_file_single_unified(
            args.input,
            args.output,
            args.model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
