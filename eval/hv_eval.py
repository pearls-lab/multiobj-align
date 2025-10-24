import os
import json
import torch
import torch.nn.functional as F
import gc
import argparse
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, pipeline
import numpy as np
from pathlib import Path
import time
from collections import defaultdict


class MultiDimensionalRewardModel:
    """
    Multi-dimensional reward model for scoring conversations across helpfulness, honesty, and truthfulness.
    MODIFIED: Uses separate GPUs for each reward model to avoid OOM.
    ADDED: Sigmoid transformation for Bradley-Terry models.
    """

    def __init__(
        self,
        helpfulness_model: str = "Jennny/llama3_8b_helpful_rm_full",
        honesty_model: str = "Jennny/llama3_8b_honest_rm_full",
        truthfulness_model: str = "Jennny/llama3_8b_truth_rm_full",
        devices: List[str] = [
            "cuda:0",
            "cuda:1",
            "cuda:2",
        ],  # Changed: Use multiple devices
        batch_size: int = 4,
        apply_sigmoid: bool = True,  # NEW: Option to apply sigmoid transformation
        temperature: float = 1.0,  # NEW: Temperature scaling for sigmoid
    ):
        """
        Initialize the multi-dimensional reward model.

        Args:
            helpfulness_model: Path/name of the helpfulness reward model
            honesty_model: Path/name of the honesty reward model
            truthfulness_model: Path/name of the truthfulness reward model
            devices: List of devices for [helpfulness, honesty, truthfulness] models
            batch_size: Batch size for processing
            apply_sigmoid: Whether to apply sigmoid transformation to raw scores
            temperature: Temperature scaling factor for sigmoid (lower = more extreme probabilities)
        """
        # Ensure we have at least 3 devices, repeat if necessary
        if len(devices) < 3:
            devices = devices * (3 // len(devices) + 1)

        self.help_device = devices[0]
        self.honesty_device = devices[1]
        self.truth_device = devices[2]

        self.batch_size = batch_size
        self.apply_sigmoid = apply_sigmoid
        self.temperature = temperature

        print(f"Initializing Multi-Dimensional Reward Models across GPUs")
        print(f"   Helpfulness: {self.help_device}")
        print(f"   Honesty: {self.honesty_device}")
        print(f"   Truthfulness: {self.truth_device}")
        print(
            f"   Sigmoid transformation: {'Enabled' if self.apply_sigmoid else 'Disabled'}"
        )
        if self.apply_sigmoid:
            print(f"   Temperature scaling: {self.temperature}")

        # Initialize helpfulness reward model
        print(f"Loading Helpfulness RM: {helpfulness_model} on {self.help_device}")
        self.help_tokenizer = AutoTokenizer.from_pretrained(helpfulness_model)
        self.help_pipe = pipeline(
            "sentiment-analysis",
            model=helpfulness_model,
            device=self.help_device,  # Use dedicated device
            tokenizer=self.help_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        # Clear cache before loading next model
        torch.cuda.empty_cache()
        gc.collect()

        # Initialize honesty reward model
        print(f"Loading Honesty RM: {honesty_model} on {self.honesty_device}")
        self.honesty_tokenizer = AutoTokenizer.from_pretrained(honesty_model)
        self.honesty_pipe = pipeline(
            "sentiment-analysis",
            model=honesty_model,
            device=self.honesty_device,  # Use dedicated device
            tokenizer=self.honesty_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        # Clear cache before loading next model
        torch.cuda.empty_cache()
        gc.collect()

        # Initialize truthfulness reward model
        print(f"Loading Truthfulness RM: {truthfulness_model} on {self.truth_device}")
        self.truth_tokenizer = AutoTokenizer.from_pretrained(truthfulness_model)
        self.truth_pipe = pipeline(
            "sentiment-analysis",
            model=truthfulness_model,
            device=self.truth_device,  # Use dedicated device
            tokenizer=self.truth_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        # Common pipeline kwargs
        self.pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",  # Get raw logits/scores
            "batch_size": self.batch_size,
        }

        print(
            f"All reward models initialized successfully across {len(set([self.help_device, self.honesty_device, self.truth_device]))} GPUs!"
        )

        # Print GPU memory usage
        self._print_gpu_memory_usage()

    def _print_gpu_memory_usage(self):
        """Print current GPU memory usage for all devices."""
        devices = [self.help_device, self.honesty_device, self.truth_device]
        unique_devices = list(set(devices))

        print("\nGPU Memory Usage:")
        for device in unique_devices:
            if device.startswith("cuda"):
                device_id = int(device.split(":")[1])
                allocated = torch.cuda.memory_allocated(device_id) / 1e9
                reserved = torch.cuda.memory_reserved(device_id) / 1e9
                print(
                    f"   {device}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )

    def _apply_sigmoid_transform(self, raw_score: float) -> float:
        """
        Apply sigmoid transformation to raw Bradley-Terry scores.

        Args:
            raw_score: Raw score from the reward model

        Returns:
            Sigmoid-transformed score between 0 and 1
        """
        if not self.apply_sigmoid:
            return raw_score

        # Convert to tensor for sigmoid operation
        score_tensor = torch.tensor(raw_score / self.temperature, dtype=torch.float32)
        sigmoid_score = F.sigmoid(score_tensor).item()

        return sigmoid_score

    def _format_conversation(self, query: str, response: str, tokenizer) -> str:
        """Format conversation for reward model evaluation."""
        chat = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]

        # Format conversation for the reward model
        formatted_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )

        # Remove BOS token if present
        if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
            formatted_text = formatted_text.replace(tokenizer.bos_token, "")

        return formatted_text

    def score_conversation(self, query: str, response: str) -> Dict[str, float]:
        """Score a single conversation across all dimensions."""
        scores = {}

        # Score helpfulness
        help_text = self._format_conversation(query, response, self.help_tokenizer)
        help_output = self.help_pipe([help_text], **self.pipe_kwargs)
        raw_help_score = help_output[0][0]["score"]
        scores["helpfulness"] = self._apply_sigmoid_transform(raw_help_score)
        scores["helpfulness_raw"] = raw_help_score  # Keep raw score for analysis

        # Score honesty
        honesty_text = self._format_conversation(
            query, response, self.honesty_tokenizer
        )
        honesty_output = self.honesty_pipe([honesty_text], **self.pipe_kwargs)
        raw_honesty_score = honesty_output[0][0]["score"]
        scores["honesty"] = self._apply_sigmoid_transform(raw_honesty_score)
        scores["honesty_raw"] = raw_honesty_score  # Keep raw score for analysis

        # Score truthfulness
        truth_text = self._format_conversation(query, response, self.truth_tokenizer)
        truth_output = self.truth_pipe([truth_text], **self.pipe_kwargs)
        raw_truth_score = truth_output[0][0]["score"]
        scores["truthfulness"] = self._apply_sigmoid_transform(raw_truth_score)
        scores["truthfulness_raw"] = raw_truth_score  # Keep raw score for analysis

        return scores

    def score_multiple_conversations(
        self, queries: List[str], responses: List[str]
    ) -> List[Dict[str, float]]:
        """Score multiple conversations efficiently in batches."""
        if len(queries) != len(responses):
            raise ValueError("Number of queries must match number of responses")

        if not queries:
            return []

        all_scores = []

        # Process in batches
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i : i + self.batch_size]
            batch_responses = responses[i : i + self.batch_size]

            print(
                f"Processing batch {i//self.batch_size + 1}/{(len(queries)-1)//self.batch_size + 1}"
            )

            # Prepare texts for each dimension
            help_texts = [
                self._format_conversation(q, r, self.help_tokenizer)
                for q, r in zip(batch_queries, batch_responses)
            ]
            honesty_texts = [
                self._format_conversation(q, r, self.honesty_tokenizer)
                for q, r in zip(batch_queries, batch_responses)
            ]
            truth_texts = [
                self._format_conversation(q, r, self.truth_tokenizer)
                for q, r in zip(batch_queries, batch_responses)
            ]

            # Score each dimension (each on its own GPU)
            try:
                help_outputs = self.help_pipe(help_texts, **self.pipe_kwargs)
                honesty_outputs = self.honesty_pipe(honesty_texts, **self.pipe_kwargs)
                truth_outputs = self.truth_pipe(truth_texts, **self.pipe_kwargs)
            except torch.cuda.OutOfMemoryError as e:
                print(
                    f"ERROR: CUDA OOM in batch {i//self.batch_size + 1}. Try reducing batch_size."
                )
                self._print_gpu_memory_usage()
                raise e

            # Combine scores with sigmoid transformation
            for help_out, honesty_out, truth_out in zip(
                help_outputs, honesty_outputs, truth_outputs
            ):
                # Extract raw scores
                raw_help_score = help_out[0]["score"]
                raw_honesty_score = honesty_out[0]["score"]
                raw_truth_score = truth_out[0]["score"]

                # Apply sigmoid transformation
                scores = {
                    "helpfulness": self._apply_sigmoid_transform(raw_help_score),
                    "honesty": self._apply_sigmoid_transform(raw_honesty_score),
                    "truthfulness": self._apply_sigmoid_transform(raw_truth_score),
                    "helpfulness_raw": raw_help_score,
                    "honesty_raw": raw_honesty_score,
                    "truthfulness_raw": raw_truth_score,
                }
                all_scores.append(scores)

            # Clear GPU cache periodically
            if i % (self.batch_size * 2) == 0:
                for device in [
                    self.help_device,
                    self.honesty_device,
                    self.truth_device,
                ]:
                    if device.startswith("cuda"):
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                gc.collect()

        return all_scores


def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    malformed_count = 0
    valid_count = 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parsed_item = json.loads(line)

                    # Ensure parsed item is a dictionary
                    if not isinstance(parsed_item, dict):
                        print(
                            f"WARNING: Line {line_num} in {filepath}: Expected dict, got {type(parsed_item)}"
                        )
                        malformed_count += 1
                        continue

                    # Check for expected fields from UltraFeedback inference
                    if "instruction" not in parsed_item and "query" not in parsed_item:
                        print(
                            f"WARNING: Line {line_num} in {filepath}: Missing instruction/query field"
                        )
                        malformed_count += 1
                        continue

                    if (
                        "model_response" not in parsed_item
                        and "full_response" not in parsed_item
                    ):
                        print(
                            f"WARNING: Line {line_num} in {filepath}: Missing response field"
                        )
                        malformed_count += 1
                        continue

                    data.append(parsed_item)
                    valid_count += 1

                except json.JSONDecodeError as e:
                    print(
                        f"WARNING: Line {line_num} in {filepath}: JSON parsing error - {e}"
                    )
                    malformed_count += 1
                    continue
                except Exception as e:
                    print(
                        f"WARNING: Line {line_num} in {filepath}: Unexpected error - {e}"
                    )
                    malformed_count += 1
                    continue

        print(f"Loaded {valid_count} valid entries from {filepath}")
        if malformed_count > 0:
            print(f"WARNING: Skipped {malformed_count} malformed lines in {filepath}")
        return data

    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return []
    except Exception as e:
        print(f"ERROR: Error loading {filepath}: {e}")
        return []


def extract_query_and_response(item: Dict) -> Tuple[str, str]:
    """Extract query and response from data item, handling different formats."""
    # Handle UltraFeedback format
    if "instruction" in item:
        query = item["instruction"]
    elif "query" in item:
        query = item["query"]
    else:
        raise ValueError("No query/instruction found in item")

    # Handle response field
    if "model_response" in item:
        response = item["model_response"]
    elif "full_response" in item:
        response = item["full_response"]
    else:
        raise ValueError("No response found in item")

    return query, response


def evaluate_model_responses(
    input_file: str,
    output_file: str,
    helpfulness_model: str,
    honesty_model: str,
    truthfulness_model: str,
    devices: List[str] = ["cuda:0", "cuda:1", "cuda:2"],  # Changed: Accept device list
    batch_size: int = 4,
    model_type: str = "unknown",
    head_index: Optional[int] = None,
    apply_sigmoid: bool = True,  # NEW: Option to apply sigmoid
    temperature: float = 1.0,  # NEW: Temperature scaling
):
    """
    Evaluate model responses across all three dimensions.

    Args:
        input_file: Path to JSONL file with model responses
        output_file: Path to output JSON file with evaluation results
        helpfulness_model: Path to helpfulness reward model
        honesty_model: Path to honesty reward model
        truthfulness_model: Path to truthfulness reward model
        devices: List of devices for [helpfulness, honesty, truthfulness] models
        batch_size: Batch size for processing
        model_type: Type of model (base/sft/dpo)
        head_index: Head index for DPO models
        apply_sigmoid: Whether to apply sigmoid transformation
        temperature: Temperature scaling for sigmoid
    """
    print(f"Starting evaluation for {input_file}")
    print(f"Model type: {model_type}, Head index: {head_index}")
    print(f"Sigmoid transformation: {'Enabled' if apply_sigmoid else 'Disabled'}")

    # Load data
    data = load_jsonl_file(input_file)
    if not data:
        print("ERROR: No valid data found. Exiting.")
        return

    # Initialize reward models with multiple GPUs and sigmoid option
    rm = MultiDimensionalRewardModel(
        helpfulness_model=helpfulness_model,
        honesty_model=honesty_model,
        truthfulness_model=truthfulness_model,
        devices=devices,
        batch_size=batch_size,
        apply_sigmoid=apply_sigmoid,
        temperature=temperature,
    )

    # Prepare queries and responses
    queries = []
    responses = []
    original_items = []

    for item in data:
        try:
            query, response = extract_query_and_response(item)
            queries.append(query)
            responses.append(response)
            original_items.append(item)
        except Exception as e:
            print(f"WARNING: Skipping item due to error: {e}")
            continue

    print(f"Processing {len(queries)} valid query-response pairs")

    # Score all conversations
    start_time = time.time()
    all_scores = rm.score_multiple_conversations(queries, responses)
    end_time = time.time()

    print(f"⏱️ Scoring completed in {end_time - start_time:.2f} seconds")

    # Prepare results
    results = []
    dimension_stats = {
        "helpfulness": {"total": 0, "sum": 0, "scores": []},
        "honesty": {"total": 0, "sum": 0, "scores": []},
        "truthfulness": {"total": 0, "sum": 0, "scores": []},
    }

    # Also track raw scores if sigmoid is applied
    if apply_sigmoid:
        raw_dimension_stats = {
            "helpfulness_raw": {"total": 0, "sum": 0, "scores": []},
            "honesty_raw": {"total": 0, "sum": 0, "scores": []},
            "truthfulness_raw": {"total": 0, "sum": 0, "scores": []},
        }

    for original_item, scores in zip(original_items, all_scores):
        # Update statistics for sigmoid-transformed scores
        for dimension in ["helpfulness", "honesty", "truthfulness"]:
            score = scores[dimension]
            dimension_stats[dimension]["total"] += 1
            dimension_stats[dimension]["sum"] += score
            dimension_stats[dimension]["scores"].append(score)

        # Update statistics for raw scores if sigmoid is applied
        if apply_sigmoid:
            for dimension in ["helpfulness_raw", "honesty_raw", "truthfulness_raw"]:
                score = scores[dimension]
                raw_dimension_stats[dimension]["total"] += 1
                raw_dimension_stats[dimension]["sum"] += score
                raw_dimension_stats[dimension]["scores"].append(score)

        # Create result entry
        result = {
            "id": original_item.get("id", f"item_{len(results)}"),
            "query": queries[len(results)],
            "response": responses[len(results)],
            "scores": scores,
            "metadata": {
                "model_type": model_type,
                "head_index": head_index,
                "sigmoid_applied": apply_sigmoid,
                "temperature": temperature if apply_sigmoid else None,
                "original_metadata": original_item.get("metadata", {}),
            },
        }
        results.append(result)

    # Calculate summary statistics for sigmoid-transformed scores
    summary_stats = {}
    for dimension in ["helpfulness", "honesty", "truthfulness"]:
        scores_list = dimension_stats[dimension]["scores"]
        if scores_list:
            summary_stats[dimension] = {
                "mean": np.mean(scores_list),
                "std": np.std(scores_list),
                "min": np.min(scores_list),
                "max": np.max(scores_list),
                "median": np.median(scores_list),
                "count": len(scores_list),
            }

    # Calculate summary statistics for raw scores if sigmoid is applied
    if apply_sigmoid:
        for dimension in ["helpfulness_raw", "honesty_raw", "truthfulness_raw"]:
            scores_list = raw_dimension_stats[dimension]["scores"]
            if scores_list:
                summary_stats[dimension] = {
                    "mean": np.mean(scores_list),
                    "std": np.std(scores_list),
                    "min": np.min(scores_list),
                    "max": np.max(scores_list),
                    "median": np.median(scores_list),
                    "count": len(scores_list),
                }

    # Prepare final output
    output_data = {
        "evaluation_metadata": {
            "input_file": input_file,
            "model_type": model_type,
            "head_index": head_index,
            "helpfulness_rm": helpfulness_model,
            "honesty_rm": honesty_model,
            "truthfulness_rm": truthfulness_model,
            "devices": devices,
            "batch_size": batch_size,
            "sigmoid_applied": apply_sigmoid,
            "temperature": temperature if apply_sigmoid else None,
            "total_samples": len(results),
            "evaluation_time": end_time - start_time,
        },
        "summary_statistics": summary_stats,
        "detailed_results": results,
    }

    # Save results
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if there's actually a directory path
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(
        f"\nEvaluation Summary for {model_type}"
        + (f" Head {head_index}" if head_index is not None else "")
    )
    print("=" * 50)

    # Print sigmoid-transformed scores
    for dimension in ["helpfulness", "honesty", "truthfulness"]:
        stats = summary_stats[dimension]
        print(
            f"{dimension.capitalize()} ({'Sigmoid-transformed' if apply_sigmoid else 'Raw'}):"
        )
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Median: {stats['median']:.4f}")
        print()

    # Print raw scores if sigmoid was applied
    if apply_sigmoid:
        print("Raw Scores (before sigmoid transformation):")
        print("-" * 40)
        for dimension in ["helpfulness_raw", "honesty_raw", "truthfulness_raw"]:
            stats = summary_stats[dimension]
            base_dim = dimension.replace("_raw", "")
            print(f"{base_dim.capitalize()} (Raw):")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Median: {stats['median']:.4f}")
            print()

    print(f"Results saved to: {output_file}")

    # Clear GPU memory on all devices
    for device in devices:
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dimensional evaluation using reward models with optional sigmoid transformation"
    )

    # Input/Output
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL file with model responses",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file for evaluation results",
    )

    # Reward Models
    parser.add_argument(
        "--helpfulness_model",
        type=str,
        default="Jennny/llama3_8b_helpful_rm_full",
        help="Helpfulness reward model",
    )
    parser.add_argument(
        "--honesty_model",
        type=str,
        default="Jennny/llama3_8b_honest_rm_full",
        help="Honesty reward model",
    )
    parser.add_argument(
        "--truthfulness_model",
        type=str,
        default="Jennny/llama3_8b_truth_rm_full",
        help="Truthfulness reward model",
    )

    # Model Info
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "sft", "dpo"],
        required=True,
        help="Type of model being evaluated",
    )
    parser.add_argument(
        "--head_index",
        type=int,
        help="Head index for DPO models (0=helpfulness, 1=honesty, 2=truthfulness)",
    )

    # Processing - MODIFIED: Support multiple devices
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2"],
        help="Devices for [helpfulness, honesty, truthfulness] models (e.g., cuda:0 cuda:1 cuda:2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing (reduced default for multi-GPU)",
    )

    # NEW: Sigmoid transformation options
    parser.add_argument(
        "--apply_sigmoid",
        action="store_true",
        default=True,
        help="Apply sigmoid transformation to Bradley-Terry scores (default: True)",
    )
    parser.add_argument(
        "--no_sigmoid",
        action="store_true",
        help="Disable sigmoid transformation (use raw Bradley-Terry scores)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for sigmoid transformation (default: 1.0)",
    )

    args = parser.parse_args()

    # Handle sigmoid flag logic
    if args.no_sigmoid:
        apply_sigmoid = False
    else:
        apply_sigmoid = args.apply_sigmoid

    # Validate devices
    if len(args.devices) < 3:
        print(
            f"WARNING: Warning: Only {len(args.devices)} devices provided. Models will share GPUs."
        )

    # Run evaluation
    evaluate_model_responses(
        input_file=args.input_file,
        output_file=args.output_file,
        helpfulness_model=args.helpfulness_model,
        honesty_model=args.honesty_model,
        truthfulness_model=args.truthfulness_model,
        devices=args.devices,
        batch_size=args.batch_size,
        model_type=args.model_type,
        head_index=args.head_index,
        apply_sigmoid=apply_sigmoid,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
