import json
import re
import argparse
import os


def extract_numbers_from_text(text):
    """Extract all numbers from text, handling currency, percentages, and comma-separated numbers."""
    # Remove commas within numbers first (e.g., "17,500" -> "17500")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    # Find all numbers including those with currency symbols and percentages
    numbers = re.findall(r"\$?(\d+(?:\.\d+)?%?)", text)

    # Clean up the numbers (remove $ and %)
    cleaned_numbers = []
    for num in numbers:
        num = num.strip()
        if num.endswith("%"):
            num = num[:-1]
        cleaned_numbers.append(num)

    return cleaned_numbers


def extract_boxed_content(text):
    """
    Extract content inside \boxed{} with proper handling of nested braces.
    Returns the content or None if no boxed content is found.
    """
    # Find all occurrences of \boxed{
    all_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not all_starts:
        return None

    # Get the last occurrence (typically the final answer)
    boxed_start = all_starts[-1]

    # Position of the opening brace
    start_pos = boxed_start + 7  # length of '\boxed{'

    # Find matching closing brace by counting
    brace_count = 1
    for i in range(start_pos, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                # Found the matching closing brace
                return text[start_pos:i]

    # If no matching closing brace found
    return None


def extract_answer(step, question):
    """
    Extract ONLY answers in \boxed{} format.
    Returns a tuple (extracted_answer, used_boxed_format) where
    extracted_answer is None if no boxed answer is found.
    """
    # Try to find answer in \boxed{} format with improved handling of nested braces
    boxed_content = extract_boxed_content(step)
    if boxed_content:
        answer = boxed_content.strip()
        # Remove commas and clean up
        answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
        answer = answer.replace("$", "").strip()
        if answer.endswith("%"):
            answer = answer[:-1]
        return answer, True

    # If no boxed content is found, return None
    return None, False


def process_response_to_steps(response):
    """
    Splits the response string into steps.
    Assumes steps are separated by double newlines.
    """
    steps = [step.strip() for step in response.split("\n\n") if step.strip()]
    return steps


def process_math_data(input_file, output_file):
    """
    Process math data from input file and save results to output file.

    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSON file
    """
    results = []
    non_boxed_cases = []  # For tracking cases where answer wasn't in \boxed{} format
    first_example = True  # For printing debug info on the first example

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    # Open the JSONL file (each line is a JSON object)
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)

            print(f"\nProcessing example {example['id']}:")
            print("Question:", example["question"])

            # Process the model response into individual steps
            response_steps = process_response_to_steps(example["model_response"])
            print("\nProcessed response steps:")
            for i, step in enumerate(response_steps, 1):
                print(f"\nStep {i}:")
                print(step)

            # Format the data (here the "system" prompt is provided for consistency)
            data = {
                "system": "Please reason step by step, and put your final answer within \\boxed{}.",
                "query": example["question"],
                "response": response_steps,
            }

            # Since there are no step_rewards in the new format, we'll set it to a default value
            # or we could calculate some other metric if needed
            # For now, let's set a default value of 1.0 for each step
            step_reward = [1.0] * len(response_steps)

            # Calculate average reward for the steps (now just 1.0)
            avg_reward = 1.0

            # Check all steps in reverse order for boxed answers
            extracted_answer = None
            used_boxed = False

            for step in reversed(response_steps):
                step_answer, is_boxed = extract_answer(step, example["question"])
                if is_boxed:
                    extracted_answer = step_answer
                    used_boxed = True
                    break

            # Important: Do NOT fall back to non-boxed extraction methods
            # extracted_answer remains None if no boxed answer was found

            if first_example:
                print("\nFirst example formatted output:")
                print(
                    json.dumps(
                        {
                            "id": example["id"],
                            "question": example["question"],
                            "response_steps": response_steps,
                            "ground_truth": example.get("ground_truth", ""),
                            "extracted_answer": extracted_answer,
                            "used_boxed_format": used_boxed,
                            "num_steps": len(data["response"]),
                        },
                        indent=2,
                    )
                )
                first_example = False

            if not used_boxed:
                non_boxed_cases.append(
                    {
                        "id": example["id"],
                        "question": example["question"],
                        "last_step": response_steps[-1] if response_steps else "",
                        "extracted_answer": None,  # Explicitly set to None
                    }
                )

            # Append the processed result in the desired format
            results.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "response_steps": response_steps,
                    "ground_truth": example.get("ground_truth", ""),
                    "extracted_answer": extracted_answer,  # Will be None if no boxed answer found
                    "used_boxed_format": used_boxed,
                    "num_steps": len(data["response"]),
                }
            )

            print(f"\nExtracted answer: {extracted_answer}")
            print(f"Used \\boxed{{}} format: {used_boxed}")
            print("-" * 80)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the results to a new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Report any cases where the answer was not found in \boxed{} format
    if non_boxed_cases:
        print("\nCases where answer was not in \\boxed{} format:")
        for case in non_boxed_cases:
            print(f"\nID: {case['id']}")
            print(f"Question: {case['question']}")
            print(f"Last step: {case['last_step']}")
            print(f"Extracted answer: None")
    else:
        print("\nAll answers were properly formatted with \\boxed{}")

    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process math problem data and extract answers from boxed format."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSONL file path with model responses",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path for processed results",
    )

    args = parser.parse_args()

    try:
        process_math_data(args.input, args.output)
        print(f"\nProcessing completed successfully!")
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
