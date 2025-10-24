import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def format_messages_with_student_for_reward_model(
    messages: List[Dict[str, str]],
) -> str:
    """Format messages_with_student for reward model scoring."""
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
    print(f"Loading engagement reward model: {model_name}")

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


def evaluate_engagement_with_reward_model(
    input_path: Path,
    output_path: Path,
    model_name: str = "Jennny/eng_rm_1e5_700",
    max_length: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
    num_samples: Optional[int] = None,
):
    """Evaluate messages_with_students using engagement reward model.

    This variant prefers the 'messages_with_student' field and falls back to
    'final_messages' when necessary.
    """

    # Load model
    model, tokenizer = load_reward_model(model_name, device)

    # Load input data
    print(f"Loading data from {input_path}")
    messages_with_students = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if num_samples is not None and len(messages_with_students) >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                # Prefer messages_with_student, fallback to final_messages
                messages_with_student_messages = row.get("messages_with_student", [])
                if not messages_with_student_messages:
                    messages_with_student_messages = row.get("final_messages", [])

                if not messages_with_student_messages:
                    print(
                        f"Warning: No messages_with_student or final_messages found in row {line_num}"
                    )
                    continue

                messages_with_students.append((row, messages_with_student_messages))

            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    if not messages_with_students:
        print("No valid messages_with_students found!")
        return

    print(f"Found {len(messages_with_students)} messages_with_students to evaluate")

    # Format messages_with_students for scoring
    formatted_messages_with_students = []
    for row, messages in messages_with_students:
        formatted_conv = format_messages_with_student_for_reward_model(messages)
        formatted_messages_with_students.append(formatted_conv)

    # Score with engagement model
    print("Scoring with engagement reward model...")
    engagement_scores = score_texts_batch(
        model, tokenizer, formatted_messages_with_students, max_length, batch_size
    )

    # Compile results
    results = []
    total_score = 0
    valid_count = 0
    error_count = 0

    for i, (row, messages) in enumerate(messages_with_students):
        score = engagement_scores[i] if i < len(engagement_scores) else float("nan")

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
            predicted_label = "ENGAGING"
        else:
            predicted_label = "NOT_ENGAGING"

        result = {
            "id": row.get("id", f"conv_{i}"),
            "predicted_label": predicted_label,
            "engagement_score": score,
            "is_error": is_error,
            "messages_with_student": messages,
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
            print(
                f"Processed {i + 1}/{len(messages_with_students)} messages_with_students"
            )

    # Calculate overall average
    overall_avg_score = total_score / valid_count if valid_count > 0 else 0

    # Write results
    print(f"Writing results to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            # Handle NaN values for JSON serialization
            if isinstance(result["engagement_score"], float) and math.isnan(
                result["engagement_score"]
            ):
                result["engagement_score"] = None
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Print summary statistics
    print(f"\nEngagement Evaluation Summary:")
    print(f"Total messages_with_students: {len(messages_with_students)}")
    print(f"Successfully scored: {valid_count}")
    print(f"Errors: {error_count}")
    print(f"Overall average engagement score: {overall_avg_score:.4f}")

    # Count predictions by label
    engaging_count = len([r for r in results if r["predicted_label"] == "ENGAGING"])
    not_engaging_count = len(
        [r for r in results if r["predicted_label"] == "NOT_ENGAGING"]
    )

    print(f"\nPrediction Summary:")
    print(f"ENGAGING: {engaging_count}")
    print(f"NOT_ENGAGING: {not_engaging_count}")
    print(f"Total valid predictions: {valid_count}")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate messages_with_students using engagement reward model "
            "(prefers 'messages_with_student', falls back to 'final_messages')"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input JSONL file with 'messages_with_student' (preferred) or 'final_messages'"
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
        default="Jennny/eng_rm_1e5_700",
        help="Hugging Face engagement reward model",
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

    evaluate_engagement_with_reward_model(
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
