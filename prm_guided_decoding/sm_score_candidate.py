#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def format_messages_with_candidate(
    messages: List[Dict[str, str]], candidate: str
) -> str:
    """Format a chat into the reward model text format."""
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"Student: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(content)

    lines.append(f"Assistant: {candidate}")
    return "\n\n".join(lines)


def load_model_and_tokenizer(model_name: str, device_id: int = None):
    """Load model on specific device."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B", trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Load model on specific GPU if device_id provided
    if device_id is not None:
        device = f"cuda:{device_id}"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            trust_remote_code=True,
        ).to(device)
    else:
        # Use device_map="auto" for model parallelism across GPUs
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer


def score_texts_gpu(
    model, tokenizer, texts: List[str], max_length: int = 2048, batch_size: int = 8
) -> List[float]:
    """Score texts on a specific GPU."""
    scores: List[float] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            # Use autocast for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(**enc)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                pos = probs[:, 1].detach().float().cpu().tolist()
                scores.extend(pos)

    return scores


def process_worker(args_tuple):
    """Worker function for processing a chunk of data on a specific GPU."""
    chunk_data, model_name, max_length, batch_size, device_id, worker_id = args_tuple

    # Initialize CUDA in this process
    torch.cuda.set_device(device_id)

    # Load model on this GPU
    model, tokenizer = load_model_and_tokenizer(model_name, device_id=device_id)

    results = []

    for row_data in chunk_data:
        messages, candidates, row = row_data

        # Prepare and score this row's candidates
        texts = [format_messages_with_candidate(messages, c) for c in candidates]
        scores = score_texts_gpu(
            model, tokenizer, texts, max_length=max_length, batch_size=batch_size
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
        }
        results.append(out_obj)

        print(
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
        print(f"Sampled {num_samples} rows from {len(data)} total (seed={seed})")

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


def process_file_parallel(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_length: int = 2048,
    batch_size: int = 8,
    num_gpus: int = 8,
    num_samples: int = None,
    seed: int = None,
):
    """Process file using multiple GPUs in parallel with real-time output."""
    print(f"Loading input data from {input_path}")
    data = load_input_data(input_path, num_samples=num_samples, seed=seed)

    if not data:
        print("No valid data found!")
        return

    print(f"Processing {len(data)} rows")
    print(f"Using {num_gpus} GPUs with batch size {batch_size}")

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
                executor.submit(process_worker, args): i
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
                        print(
                            f"[Completed {completed_count}/{len(data)}] Worker {worker_id} | "
                            f"best_idx={result['best_index']} best_score={result['best_score']:.4f} "
                            f"id={result.get('id', '')}"
                        )

                    print(f"Worker {worker_id} finished with {len(results)} results")

                except Exception as exc:
                    print(f"Worker {worker_id} generated an exception: {exc}")

    print(f"Completed! Scored {completed_count} rows → {output_path}")


def process_file_single_model_parallel(
    input_path: Path,
    output_path: Path,
    model_name: str,
    max_length: int = 2048,
    batch_size: int = 16,
    num_samples: int = None,
    seed: int = None,
):
    """Alternative: Load one model across all GPUs and process with larger batches."""
    print(f"Loading model {model_name} across all available GPUs")
    model, tokenizer = load_model_and_tokenizer(model_name)  # Uses device_map="auto"

    print(f"Loading input data from {input_path}")
    data = load_input_data(input_path, num_samples=num_samples, seed=seed)

    if not data:
        print("No valid data found!")
        return

    print(f"Processing {len(data)} rows with batch size {batch_size}")

    written = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for i, (messages, candidates, row) in enumerate(data):
            # Prepare and score this row's candidates
            texts = [format_messages_with_candidate(messages, c) for c in candidates]
            scores = score_texts_gpu(
                model, tokenizer, texts, max_length=max_length, batch_size=batch_size
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
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()  # Force write to disk immediately
            written += 1

            print(
                f"[Completed {written}/{len(data)}] best_idx={best_idx} best_score={best_score:.4f} id={row.get('id', '')}"
            )

    print(f"Completed! Scored {written} rows → {output_path}")


def main():
    # Fix for CUDA multiprocessing issue
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Score preference candidates with reward model using multiple GPUs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with candidate responses",
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
        default="Jennny/eng_rm_1e5_350",
        help="Hugging Face reward model repo",
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
        print("CUDA not available!")
        return

    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    if args.num_gpus > available_gpus:
        print(
            f"Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}."
        )
        args.num_gpus = available_gpus

    if args.strategy == "multi_process":
        process_file_parallel(
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
        process_file_single_model_parallel(
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
