#!/usr/bin/env python3
import argparse
import json
import os
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


def resolve_best_candidate(row: Dict[str, Any]) -> Optional[str]:
    if isinstance(row.get("best_candidate"), str):
        return row["best_candidate"]
    candidates = row.get("candidates") or []
    best_index = row.get("best_index")
    if isinstance(candidates, list) and isinstance(best_index, int):
        if 0 <= best_index < len(candidates):
            return candidates[best_index]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Following the chosen best assistant candidate, generate the next student response (1 message) using GPT-4o"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input scored JSONL with fields: messages, best_candidate (or candidates + best_index)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
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
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    total_written = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(iter_jsonl(args.input)):
            if args.num_samples is not None and total_written >= args.num_samples:
                break

            messages: List[Dict[str, str]] = row.get("messages", [])
            if not isinstance(messages, list) or len(messages) == 0:
                continue

            best_candidate = resolve_best_candidate(row)
            if not best_candidate:
                continue

            new_messages = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    new_messages.append({"role": "user", "content": msg.get("content")})
                else:
                    new_messages.append(
                        {"role": "assistant", "content": msg.get("content")}
                    )

            # Append the chosen assistant candidate
            convo_plus = [*new_messages, {"role": "user", "content": best_candidate}]
            convo_plus_record = [
                *messages,
                {"role": "assistant", "content": best_candidate},
            ]

            system_prompt = """You are a student who is learning programming as a beginner with a tutor. Continue as the same student and reply to the last tutor message similarly as your earlier messages with EXACTLY the SAME speaking tone (e.g., curious, impatient, informal, etc.), response style (e.g., short, long, incomplete, etc.), amount of discourse marker (e.g., not using any discourse markers), understanding level (e.g., making mistakes), and engagement level (e.g., less engaged in the session)."""

            # Generate the next student response
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    *convo_plus,
                ],
                temperature=args.temp,
                top_p=args.top_p,
                seed=args.seed,
            )
            student_text = (resp.choices[0].message.content or "").strip()

            out_obj = {
                "id": row.get("id"),
                "cut_index": row.get("cut_index"),
                "rating": row.get("rating"),
                "model": args.model,
                "temperature": args.temp,
                "top_p": args.top_p,
                "seed": args.seed,
                "best_candidate": best_candidate,
                "student_response": student_text,
                "messages_with_student": [
                    *convo_plus_record,
                    {"role": "user", "content": student_text},
                ],
            }

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            total_written += 1

            print(
                f"[Progress] wrote student response for id={row.get('id', '')} ({total_written})"
            )

    print(f"Done. Wrote {total_written} rows → {args.output}")


if __name__ == "__main__":
    main()
