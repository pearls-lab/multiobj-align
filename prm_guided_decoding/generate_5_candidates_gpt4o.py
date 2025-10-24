#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI


def load_prefixes(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate 5 assistant candidates per prefix using GPT-4o"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL of prefixes produced by prepare_preference_prefixes.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL with 5 candidates per prefix",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--seed_base", type=int, default=1234)
    parser.add_argument(
        "--skip", type=int, default=0, help="Number of entries to skip (for resuming)"
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
    )

    prefixes = load_prefixes(args.input)
    total = len(prefixes)

    # Skip already processed entries and open in append mode if resuming
    if args.skip > 0:
        prefixes = prefixes[args.skip :]
        print(
            f"Skipping first {args.skip} entries, processing {len(prefixes)} remaining entries"
        )
        open_mode = "a"
    else:
        open_mode = "w"

    with args.output.open(open_mode, encoding="utf-8") as out_f:
        for idx, row in enumerate(prefixes):
            messages = row.get("messages", [])
            if not isinstance(messages, list) or not messages:
                continue

            # Ensure last message is from user so model generates assistant
            if messages[-1].get("role") != "user":
                continue
            system_prompt = """You are a tutor who is helping a beginner student learn programming. Continue as the same tutor and reply similarly to the last student message, matching EXACTLY the SAME speaking tone and tutoring style as in your earlier messages (e.g. reply to the student's last message concisely in 1-2 sentences and then always ask a meaningful follow-up question)."""

            candidates: List[str] = []
            for k in range(args.num_candidates):
                seed = args.seed_base + k
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        *messages,
                    ],
                    temperature=args.temp,
                    top_p=args.top_p,
                    seed=seed,
                )
                text = resp.choices[0].message.content
                candidates.append(text)

            out = {
                "id": row.get("id"),
                "cut_index": row.get("cut_index"),
                "rating": row.get("rating"),
                "messages": messages,
                "model": args.model,
                "num_candidates": args.num_candidates,
                "temperature": args.temp,
                "top_p": args.top_p,
                "candidates": candidates,
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()

            # Progress display
            actual_idx = idx + args.skip + 1
            print(
                f"[Progress] {actual_idx}/{total} prefixes processed → wrote {len(candidates)} candidates"
            )

    print(f"Wrote candidates for {len(prefixes)} prefixes to {args.output}")


if __name__ == "__main__":
    main()
