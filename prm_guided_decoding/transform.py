#!/usr/bin/env python3
"""
Transform DPO file format to preference format.
Parses the 'question' field to extract conversation messages and converts
model_candidates to candidates array.

Usage:
    python transform.py <dpo_file> <output_file>

Example:
    python transform.py dpo_qwen25_sm_ens_wt_sft_all_5candidates.jsonl output.jsonl
"""

import json
import sys
import os
import argparse
import re
from typing import Dict, List, Any


def parse_conversation(question_text: str) -> List[Dict[str, str]]:
    """
    Parse the question text to extract conversation messages.
    The question field contains alternating Assistant/Student messages.
    """
    messages = []

    # Split the text into segments starting with Assistant: or Student:
    # Use positive lookahead to keep the markers
    segments = re.split(r"\n\n(?=(?:Assistant:|Student:))", question_text)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Check if it starts with Assistant: or Student:
        if segment.startswith("Assistant:"):
            role = "assistant"
            content = segment[len("Assistant:") :].strip()
        elif segment.startswith("Student:"):
            role = "user"  # Map 'student' to 'user' for standard chat format
            content = segment[len("Student:") :].strip()
        else:
            continue

        if content:
            messages.append({"role": role, "content": content})

    return messages


def transform_dpo_to_preference(dpo_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform DPO format to preference format."""
    transformed = []

    for idx, record in enumerate(dpo_data):
        try:
            record_id = record.get("id")
            question = record.get("question", "")
            model_candidates = record.get("model_candidates", [])

            # Parse the question field into messages
            messages = parse_conversation(question)

            # Convert model_candidates to candidates format
            candidates = []
            for candidate in model_candidates:
                if "response" in candidate:
                    candidates.append(candidate["response"])

            # Create the new record in preference format
            new_record = {
                "id": record_id,
                "messages": messages,
                "candidates": candidates,
            }

            # Preserve metadata fields if present
            metadata_fields = [
                "cut_index",
                "rating",
                "model",
                "num_candidates",
                "temperature",
                "top_p",
            ]
            for field in metadata_fields:
                if field in record:
                    new_record[field] = record[field]

            transformed.append(new_record)

        except Exception as e:
            print(
                f"Warning: Error processing record {idx} (id: {record.get('id', 'unknown')}): {e}"
            )
            continue

    print(f"Successfully transformed {len(transformed)} records")
    return transformed


def load_jsonl(filename: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue
        print(f"Loaded {len(data)} records from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)


def save_jsonl(data: List[Dict[str, Any]], filename: str):
    """Save data to JSONL file."""
    try:
        # Create directory if it doesn't exist
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} records to {filename}")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Transform DPO file format to preference format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transform.py dpo_file.jsonl output.jsonl
  python transform.py data/dpo.jsonl results/transformed.jsonl
        """,
    )

    parser.add_argument("dpo_file", help="Path to the DPO JSONL file to transform")
    parser.add_argument(
        "output_file", help="Path for the output transformed JSONL file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.dpo_file):
        print(f"Error: Input file not found: {args.dpo_file}")
        sys.exit(1)

    print("=" * 60)
    print("DPO to Preference Format Transformation")
    print("=" * 60)
    print(f"Input file: {args.dpo_file}")
    print(f"Output file: {args.output_file}")
    print()

    # Load DPO file
    print("1. Loading DPO file...")
    dpo_data = load_jsonl(args.dpo_file)

    # Transform to preference format
    print("\n2. Transforming to preference format...")
    print("   - Parsing conversation messages")
    print("   - Converting model candidates")
    transformed_data = transform_dpo_to_preference(dpo_data)

    # Save the result
    print("\n3. Saving transformed data...")
    save_jsonl(transformed_data, args.output_file)

    print("\n" + "=" * 60)
    print("✅ Transformation Complete!")
    print("=" * 60)
    print(f"Original records: {len(dpo_data)}")
    print(f"Transformed records: {len(transformed_data)}")
    print(f"Success rate: {len(transformed_data)/len(dpo_data)*100:.1f}%")
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
