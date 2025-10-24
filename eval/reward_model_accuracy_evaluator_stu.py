import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def format_conversation_for_reward_model(messages: List[Dict[str, str]]) -> str:
    """Format conversation for reward model scoring."""
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"Student: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n\n".join(lines)


def load_reward_model(model_name: str, device: str = "cuda"):
    """Load reward model and tokenizer."""
    print(f"Loading accuracy reward model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        print("Warning: Could not load tokenizer, falling back to Llama tokenizer")
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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model, tokenizer


def score_texts_batch(
    model, tokenizer, texts: List[str], max_length: int = 2048, batch_size: int = 8
) -> List[float]:
    """Score texts using reward model."""
    scores: List[float] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                with torch.cuda.amp.autocast():
                    outputs = model(**enc)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    # Use positive class probability (index 1)
                    pos_scores = probs[:, 1].detach().float().cpu().tolist()
                    scores.extend(pos_scores)

            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add NaN scores for failed batches
                scores.extend([float("nan")] * len(batch_texts))

    return scores


def evaluate_accuracy_with_reward_model(
    input_path: Path,
    output_path: Path,
    model_name: str = "Jennny/acc_rm_5e6_450",
    max_length: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
    num_samples: Optional[int] = None,
):
    """Evaluate conversations using accuracy reward model.

    This variant prefers the 'conversation' field and falls back to
    'messages_with_student' when necessary.
    """

    # Load model
    model, tokenizer = load_reward_model(model_name, device)

    # Load input data
    print(f"Loading data from {input_path}")
    conversations = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if num_samples is not None and len(conversations) >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                # Prefer conversation, fallback to messages_with_student
                conversation_messages = row.get("conversation", [])
                if not conversation_messages:
                    conversation_messages = row.get("messages_with_student", [])

                if not conversation_messages:
                    print(
                        f"Warning: No conversation or messages_with_student found in row {line_num}"
                    )
                    continue

                conversations.append((row, conversation_messages))

            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    if not conversations:
        print("No valid conversations found!")
        return

    print(f"Found {len(conversations)} conversations to evaluate")

    # Format conversations for scoring
    formatted_conversations = []
    for row, messages in conversations:
        formatted_conv = format_conversation_for_reward_model(messages)
        formatted_conversations.append(formatted_conv)

    # Score with accuracy model
    print("Scoring with accuracy reward model...")
    accuracy_scores = score_texts_batch(
        model, tokenizer, formatted_conversations, max_length, batch_size
    )

    # Compile results
    results = []
    total_score = 0
    valid_count = 0
    error_count = 0

    for i, (row, messages) in enumerate(conversations):
        score = accuracy_scores[i] if i < len(accuracy_scores) else float("nan")

        is_error = math.isnan(score)
        if is_error:
            error_count += 1
        else:
            total_score += score
            valid_count += 1

        # Map score to label for compatibility
        if is_error:
            predicted_label = "NaN"
        elif score >= 0.5:
            predicted_label = "ACCURATE"
        else:
            predicted_label = "INACCURATE"

        result = {
            "id": row.get("id", f"conv_{i}"),
            "predicted_label": predicted_label,
            "accuracy_score": score,
            "is_error": is_error,
            "conversation": messages,
        }

        # Include original metadata
        for key in [
            "rating",
            "cut_index",
            "model",
            "temperature",
            "top_p",
            "seed",
            "model_response",
            "student_response",
        ]:
            if key in row:
                result[key] = row[key]

        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(conversations)} conversations")

    # Calculate overall average
    overall_avg_score = total_score / valid_count if valid_count > 0 else 0

    # Write results
    print(f"Writing results to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            # Handle NaN values for JSON serialization
            if isinstance(result["accuracy_score"], float) and math.isnan(
                result["accuracy_score"]
            ):
                result["accuracy_score"] = None
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Print summary statistics
    print(f"\nAccuracy Evaluation Summary:")
    print(f"Total conversations: {len(conversations)}")
    print(f"Successfully scored: {valid_count}")
    print(f"Errors: {error_count}")
    print(f"Overall average accuracy score: {overall_avg_score:.4f}")

    # Count predictions by label
    accurate_count = len([r for r in results if r["predicted_label"] == "ACCURATE"])
    inaccurate_count = len([r for r in results if r["predicted_label"] == "INACCURATE"])

    print(f"\nPrediction Summary:")
    print(f"ACCURATE: {accurate_count}")
    print(f"INACCURATE: {inaccurate_count}")
    print(f"Total valid predictions: {valid_count}")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate conversations using accuracy reward model "
            "(prefers 'conversation', falls back to 'messages_with_student')"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input JSONL file with 'conversation' (preferred) or 'messages_with_student'"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL file with scores",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Jennny/acc_rm_5e6_450",
        help="Hugging Face accuracy reward model",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for scoring"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    evaluate_accuracy_with_reward_model(
        input_path=args.input,
        output_path=args.out,
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
