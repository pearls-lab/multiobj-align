import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

from openai import OpenAI


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Add model_response as next assistant turn and generate the next student response using GPT-4o"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/Users/jennyshen/Desktop/Socratic/test_ensembel_qwen_5samples_with_messages.jsonl"
        ),
        help="Input JSONL with fields: messages, model_response",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/jennyshen/Desktop/Socratic/student_next_responses_from_messages.jsonl"
        ),
        help="Output JSONL with appended student response",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional: only process the first N rows (useful for testing)",
    )
    parser.add_argument(
        "--random_sample",
        type=int,
        default=None,
        help="Optional: randomly sample N rows from the input (overrides --num_samples)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    # Read all data first
    all_data = list(iter_jsonl(args.input))
    print(f"Loaded {len(all_data)} rows from {args.input}")

    # Determine which rows to process
    if args.random_sample is not None:
        # Random sampling
        if args.random_sample > len(all_data):
            print(
                f"Warning: Requested {args.random_sample} samples but only {len(all_data)} available. Using all data."
            )
            data_to_process = all_data
        else:
            print(
                f"Randomly sampling {args.random_sample} rows with seed {args.random_seed}"
            )
            random.seed(args.random_seed)
            data_to_process = random.sample(all_data, args.random_sample)
    elif args.num_samples is not None:
        # Sequential sampling (first N)
        print(f"Processing first {args.num_samples} rows")
        data_to_process = all_data[: args.num_samples]
    else:
        # Process all data
        print("Processing all rows")
        data_to_process = all_data

    total_written = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(data_to_process):

            messages: List[Dict[str, str]] = row.get("messages", [])
            model_response: str = row.get("model_response", "")

            if not isinstance(messages, list) or len(messages) == 0:
                continue

            if not model_response:
                continue

            # Add the model_response as the next assistant turn
            messages_with_assistant = [
                *messages,
                {"role": "assistant", "content": model_response},
            ]

            # Create conversation for GPT-4o (flip roles: assistant->user, user->assistant)
            gpt_conversation = []
            for msg in messages_with_assistant:
                if msg.get("role") == "assistant":
                    gpt_conversation.append(
                        {"role": "user", "content": msg.get("content")}
                    )
                else:  # user/student
                    gpt_conversation.append(
                        {"role": "assistant", "content": msg.get("content")}
                    )

            system_prompt = """You are a student who is learning programming as a beginner with a tutor. Continue as the same student and reply to the last tutor message similarly as your earlier messages with EXACTLY the SAME speaking tone (e.g., curious, impatient, informal, etc.), response style (e.g., short, long, incomplete, etc.), amount of discourse marker (e.g., not using any discourse markers), understanding level (e.g., making mistakes), and engagement level (e.g., less engaged in the session)."""

            # Generate the next student response
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    *gpt_conversation,
                ],
                temperature=args.temp,
                top_p=args.top_p,
                seed=args.seed,
            )
            student_text = (resp.choices[0].message.content or "").strip()

            # Create final messages with the new student response
            final_messages = [
                *messages_with_assistant,
                {"role": "user", "content": student_text},
            ]

            out_obj = {
                "id": row.get("id"),
                "model": args.model,
                "temperature": args.temp,
                "top_p": args.top_p,
                "seed": args.seed,
                "random_seed": args.random_seed,
                "random_sample": args.random_sample,
                "model_response": model_response,
                "student_response": student_text,
                "original_messages": messages,
                "messages_with_model_response": messages_with_assistant,
                "final_messages": final_messages,
            }

            # Preserve any additional fields from the original row
            for key, value in row.items():
                if key not in out_obj:
                    out_obj[key] = value

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            total_written += 1

            print(
                f"[Progress] wrote student response for id={row.get('id', '')} ({total_written}/{len(data_to_process)})"
            )

    print(f"Done. Wrote {total_written} rows → {args.output}")
    if args.random_sample is not None:
        print(
            f"Random sampling: {args.random_sample} samples with seed {args.random_seed}"
        )
    elif args.num_samples is not None:
        print(f"Sequential sampling: first {args.num_samples} samples")
    else:
        print("Processed all available data")


if __name__ == "__main__":
    main()
