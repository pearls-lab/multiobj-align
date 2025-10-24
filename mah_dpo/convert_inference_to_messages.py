import argparse
import json
import re
from pathlib import Path

ROLE_PATTERN = re.compile(r"(?:^|\n)(Assistant|Student):\s*", re.IGNORECASE)


def question_to_messages(question_text: str):
    """Parse a dialogue string like "Student: ...\n\nAssistant: ...\n\n..."
    into messages=[{role: 'user'|'assistant', content: str}, ...].
    Trailing empty turns (e.g., final 'Assistant:' with no content) are dropped.

    This function reverses the process of format_dialogue_as_question which joins
    messages with exactly "\n\n" separator. Handles cases where original content
    contains trailing newlines by looking for role transitions.
    """
    text = question_text or ""
    if not text.strip():
        return []

    messages = []

    # Find all role transition positions
    role_positions = []

    # Look for "\n\nStudent:" and "\n\nAssistant:" patterns
    for match in re.finditer(r"\n\n(Student|Assistant):", text):
        role_positions.append(
            (match.start() + 2, match.group(1))
        )  # +2 to skip the "\n\n"

    # Handle the first role if text starts with "Student:" or "Assistant:"
    if text.startswith("Student:"):
        role_positions.insert(0, (0, "Student"))
    elif text.startswith("Assistant:"):
        role_positions.insert(0, (0, "Assistant"))

    # Extract content for each role
    for i, (pos, role) in enumerate(role_positions):
        # Determine content start (after "Role: ")
        role_prefix_len = len(role) + 2  # "Student: " or "Assistant: "
        content_start = pos + role_prefix_len

        # Determine content end (next role position or end of text)
        if i + 1 < len(role_positions):
            content_end = role_positions[i + 1][0]
        else:
            content_end = len(text)

        # Extract content
        content = text[content_start:content_end]

        # Remove trailing "\n\n" if it exists (it connects to next role)
        if content.endswith("\n\n"):
            content = content[:-2]

        # Map role and add to messages if content is not empty
        mapped_role = "user" if role == "Student" else "assistant"
        if content:  # Only add non-empty content
            messages.append({"role": mapped_role, "content": content})

    return messages


def convert_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            # Skip if row is not a dictionary (e.g., float, string, etc.)
            if not isinstance(row, dict):
                continue

            question = row.get("question", "")
            messages = question_to_messages(question)
            # Attach messages while preserving original fields
            row["messages"] = messages
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert inference JSONL (question string) to messages array"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    args = parser.parse_args()

    convert_file(args.input, args.output)
    print(f"Wrote converted file to: {args.output}")


if __name__ == "__main__":
    main()
