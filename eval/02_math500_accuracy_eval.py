import json
import re
import random
import argparse
import os
from datasets import load_dataset


def load_math500_sample():
    """
    Load MATH-500 dataset (test split) and return a dictionary mapping problem text to answer.
    """
    ds = load_dataset("HuggingFaceH4/MATH-500")
    test_data = ds["test"]

    math500_dict = {}
    for ex in test_data:
        question_text = ex["problem"].strip()
        gt_answer = ex["answer"].strip()
        math500_dict[question_text] = gt_answer

    return math500_dict


def load_json_results(json_path):
    """
    Load the JSON file (assumed to be a list of objects) that contains, among others,
    the keys "question" and "response_steps".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_boxed_answer(response_steps):
    """
    Extract the answer enclosed in \boxed{} from the response steps.
    Returns the content inside the last \boxed{} found, or None if no boxed answer is found.
    """
    # Join all response steps to search through the entire response
    full_response = " ".join(response_steps)

    # Find all boxed answers using regex
    boxed_matches = re.findall(
        r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", full_response
    )

    if boxed_matches:
        # Return the last boxed answer found (typically the final answer)
        return boxed_matches[-1].strip()

    return None


def normalize_latex(latex_str):
    """
    Normalize LaTeX expressions for better comparison:
    - Remove \text{} wrappers
    - Normalize spaces in tuples and ordered pairs
    - Normalize different fraction formats
    - Normalize subscripts
    - Normalize vector/matrix representations
    - Normalize multiple choice answers (A) vs A
    """
    if latex_str is None:
        return None

    # Make a copy of the original string for comparison later
    original = latex_str

    # Remove \text{} wrappers
    latex_str = re.sub(r"\\text\{([^{}]+)\}", r"\1", latex_str)

    # Normalize multiple choice answers: remove parentheses around single letters
    # This handles both (A) and A format
    latex_str = re.sub(r"^\(([A-F])\)$", r"\1", latex_str)

    # Normalize spaces in tuples/pairs
    latex_str = re.sub(r"\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", r"(\1,\2)", latex_str)

    # Normalize fractions: \frac43 -> \frac{4}{3}
    latex_str = re.sub(r"\\frac(\d+)(\d+)", r"\\frac{\1}{\2}", latex_str)

    # Normalize subscripts: a_5 -> a_{5} and a_{5} -> a_5 for single digit/character subscripts
    latex_str = re.sub(r"_(\d|[a-zA-Z])(?!\{)", r"_{\1}", latex_str)
    latex_str = re.sub(r"_\{(\d|[a-zA-Z])\}", r"_\1", latex_str)

    # Convert -1/3 and \frac{-1}{3} formats to be consistent
    latex_str = re.sub(r"-\\frac\{(\d+)\}", r"\\frac{-\1}", latex_str)

    # Replace / with \frac in vectors/matrices
    if "\\begin{pmatrix}" in latex_str:
        # Convert a/b to \frac{a}{b} in matrices
        latex_str = re.sub(r"(\d+)/(\d+)", r"\\frac{\1}{\2}", latex_str)

    return latex_str


def compare_latex_answers(answer1, answer2):
    """
    Compare two LaTeX answers for equivalence, considering various formats.
    Returns True if they're equivalent, False otherwise.
    """
    # Check if answers are numerically equivalent
    val1 = normalize_latex_fraction(answer1)
    val2 = normalize_latex_fraction(answer2)

    # If both answers can be converted to numbers, compare them numerically
    if val1 is not None and val2 is not None:
        if abs(val1 - val2) < 1e-6:
            return True

    # If not numerically equivalent, try string normalization and comparison
    norm1 = normalize_latex(answer1)
    norm2 = normalize_latex(answer2)

    # Check for multiple choice answers (make case insensitive for letters A-F)
    if re.match(r"^[A-F]$", norm1, re.IGNORECASE) and re.match(
        r"^[A-F]$", norm2, re.IGNORECASE
    ):
        return norm1.upper() == norm2.upper()

    # Direct string comparison after normalization
    if norm1 == norm2:
        return True

    # Special handling for vectors/matrices
    if "\\begin{pmatrix}" in norm1 and "\\begin{pmatrix}" in norm2:
        # Extract entries and compare them individually
        entries1 = re.findall(
            r"\\\\|\\frac\{[^{}]+\}\{[^{}]+\}|\d+/\d+|[^\\\\]+", norm1
        )
        entries2 = re.findall(
            r"\\\\|\\frac\{[^{}]+\}\{[^{}]+\}|\d+/\d+|[^\\\\]+", norm2
        )

        if len(entries1) == len(entries2):
            all_match = True
            for e1, e2 in zip(entries1, entries2):
                if not compare_latex_answers(e1.strip(), e2.strip()):
                    all_match = False
                    break
            if all_match:
                return True

    return False


def normalize_latex_fraction(latex_str):
    """
    Convert LaTeX fractions to decimal values.
    Examples:
    - \frac{2}{3} -> 2/3 -> 0.6666...
    - -\frac{323}{9} -> -323/9 -> -35.8888...
    """
    if latex_str is None:
        return None

    # Handle negative fractions (with the minus sign outside the fraction)
    negative = False
    if latex_str.startswith("-"):
        negative = True
        latex_str = latex_str[1:]

    # Normalize \frac43 format to \frac{4}{3}
    latex_str = re.sub(r"\\frac(\d+)(\d+)", r"\\frac{\1}{\2}", latex_str)

    # Check for fractions in the format \frac{numerator}{denominator}
    frac_match = re.match(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", latex_str)
    if frac_match:
        try:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            result = numerator / denominator
            return -result if negative else result
        except ValueError:
            pass

    # Check for simple a/b fraction format
    frac_match = re.match(r"(\d+)/(\d+)", latex_str)
    if frac_match:
        try:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            result = numerator / denominator
            return -result if negative else result
        except ValueError:
            pass

    # If it's not a fraction or conversion failed, try the original normalization
    return normalize_numeric(latex_str)


def normalize_numeric(answer_str):
    """
    Remove commas, dollar signs, and whitespace from the answer string,
    and try to convert it to a float. If conversion fails, try to extract the first
    numeric substring and convert that to a float. If all conversion attempts fail,
    return None.
    """
    if answer_str is None:
        return None

    # Skip normalization for multiple choice answers
    if re.match(r"^[A-F]$", answer_str) or re.match(r"^\([A-F]\)$", answer_str):
        return None

    # Check for tuple format (a,b)
    tuple_match = re.match(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", answer_str)
    if tuple_match:
        try:
            # For tuples, we'll just return the first value as a numeric representation
            # This is just for potential numeric comparison and not for actual tuple comparison
            return float(tuple_match.group(1))
        except ValueError:
            pass

    # Remove common punctuation and symbols
    s = answer_str.replace(",", "").replace("$", "").strip()

    # Remove subscripts for numeric conversion (e.g., 4210_5 -> 4210)
    s = re.sub(r"_\{?[^}]*\}?", "", s)

    try:
        return float(s)
    except ValueError:
        # Try to extract the first numeric value using a regex
        match = re.search(r"([-+]?\d*\.?\d+)", s)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        else:
            return None


def compute_accuracy_by_question(json_results, math500_lookup):
    """
    Compare the extracted answer from response_steps with the ground truth answer
    in the math500_lookup dictionary (keyed by question).
    Modified to check for pre-extracted answers and handle missing boxed answers.
    """
    total_questions = 0
    correct = 0
    wrong_examples = []
    unfound_boxed_answers = []
    correct_examples = []
    questions_not_in_lookup = []

    for res in json_results:
        question = res.get("question", "").strip()

        # First check if the json_results have the ground_truth field
        json_gt = res.get("ground_truth", "").strip()

        # If question not in lookup and we have json_gt, use that
        if question not in math500_lookup:
            if json_gt:
                math500_lookup[question] = json_gt
            else:
                questions_not_in_lookup.append(question)
                continue

        total_questions += 1

        # First check if the response already has an extracted_answer field
        extracted_answer = res.get("extracted_answer")

        # If not, try to extract it from response_steps
        if not extracted_answer:
            response_steps = res.get("response_steps", [])
            extracted_answer = extract_boxed_answer(response_steps)

        # Track cases where no boxed answer is found - count as incorrect
        if not extracted_answer:
            unfound_boxed_answers.append(
                {
                    "question": question,
                    "response_steps": res.get("response_steps", []),
                    "ground_truth": math500_lookup[question],
                }
            )
            continue

        gt_ans = math500_lookup[question]

        # Try advanced LaTeX comparison first
        is_correct = compare_latex_answers(extracted_answer, gt_ans)

        # Get numerical values for reporting
        extracted_ans_num = normalize_latex_fraction(extracted_answer)
        gt_ans_num = normalize_latex_fraction(gt_ans)

        if is_correct:
            correct += 1
            correct_examples.append(
                {
                    "question": question,
                    "extracted_answer": extracted_answer,
                    "ground_truth": gt_ans,
                    "extracted_value": extracted_ans_num,
                    "ground_truth_value": gt_ans_num,
                    "normalized_extracted": normalize_latex(extracted_answer),
                    "normalized_ground_truth": normalize_latex(gt_ans),
                    "response_steps": res.get("response_steps", []),
                }
            )
        else:
            wrong_examples.append(
                {
                    "question": question,
                    "extracted_answer": extracted_answer,
                    "ground_truth": gt_ans,
                    "extracted_value": extracted_ans_num,
                    "ground_truth_value": gt_ans_num,
                    "normalized_extracted": normalize_latex(extracted_answer),
                    "normalized_ground_truth": normalize_latex(gt_ans),
                    "response_steps": res.get("response_steps", []),
                }
            )

    # Calculate accuracy based on total questions in lookup that were found in results
    accuracy = correct / total_questions if total_questions > 0 else 0.0

    if questions_not_in_lookup:
        print(
            f"Warning: {len(questions_not_in_lookup)} questions not found in MATH500 sample"
        )

    return (
        accuracy,
        total_questions,
        correct,
        correct_examples,
        wrong_examples,
        unfound_boxed_answers,
        questions_not_in_lookup,
    )


def save_evaluation_results(results_dict, output_path):
    """
    Save all evaluation results to a JSON file.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)


def evaluate_math_results(input_file, output_file):
    """
    Evaluate math results from input file and save evaluation to output file.

    Args:
        input_file (str): Path to input JSON file with processed math results
        output_file (str): Path to output JSON file for evaluation results
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    print(f"Loading MATH-500 dataset...")
    math500_lookup = load_math500_sample()

    print(f"Loading results from {input_file}...")
    json_results = load_json_results(input_file)

    print("Computing accuracy...")
    (
        accuracy,
        total_questions,
        total_correct,
        correct_examples,
        wrong_examples,
        unfound_boxed_answers,
        questions_not_in_lookup,
    ) = compute_accuracy_by_question(json_results, math500_lookup)

    # Prepare complete results dictionary
    results_dict = {
        "statistics": {
            "accuracy": accuracy,
            "accuracy_percentage": f"{accuracy*100:.2f}%",
            "total_questions_in_lookup": total_questions,
            "total_correct": total_correct,
            "total_incorrect": len(wrong_examples),
            "total_missing_boxed": len(unfound_boxed_answers),
            # The revised total incorrect includes both wrong examples and missing boxed answers
            "total_revised_incorrect": len(wrong_examples) + len(unfound_boxed_answers),
            "questions_not_in_lookup": len(questions_not_in_lookup),
            "total_examples_in_results": len(json_results),
        },
        "correct_examples": correct_examples,
        "incorrect_examples": wrong_examples,
        "unfound_boxed_answers": unfound_boxed_answers,
        "questions_not_in_lookup": questions_not_in_lookup,
    }

    # Save all results to file
    print(f"Saving evaluation results to {output_file}...")
    save_evaluation_results(results_dict, output_file)

    # Print summary statistics
    print(f"\nEvaluation Results:")
    print(f"Accuracy (including missing boxed as incorrect): {accuracy*100:.2f}%")
    print(f"Total questions found in lookup: {total_questions}")
    print(f"Total correct: {total_correct}")
    print(f"Number of incorrect answers: {len(wrong_examples)}")
    print(
        f"Number of examples with no boxed answer found (counted as incorrect): {len(unfound_boxed_answers)}"
    )
    print(
        f"Total incorrect (wrong + missing boxed): {len(wrong_examples) + len(unfound_boxed_answers)}"
    )
    print(f"Questions not found in lookup: {len(questions_not_in_lookup)}")
    print(f"Total examples in results file: {len(json_results)}")
    print(f"All evaluation results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math problem solving results by comparing extracted answers with ground truth."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input JSON file path with processed math results",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path for evaluation results",
    )

    args = parser.parse_args()

    try:
        evaluate_math_results(args.input, args.output)
        print(f"\nEvaluation completed successfully!")
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
