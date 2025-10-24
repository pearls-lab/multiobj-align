import os
import json
import re
import torch
import gc
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class VLLMEngagementJudge:
    """Evaluates math solutions for engagement using vLLM for speedup"""

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-72B-Instruct",
        tensor_parallel_size=8,
        dtype="auto",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    ):
        """Setup the model and tokenizer using vLLM"""
        print(f"Loading judge model {model_name} with vLLM...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize with vLLM instead of HuggingFace Transformers
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        # Configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.1,  # Low temperature for more deterministic outputs
            max_tokens=512,
            top_p=1.0,
        )

        print(f"Judge model loaded successfully")

    def create_judge_prompt(self, problem, solution):
        """Create a prompt for the LLM-as-judge to evaluate solution engagingness"""

        system_prompt = """You are an expert evaluator of mathematical explanations. Your task is to determine whether the mathematical content presented is ENGAGING or NOT ENGAGING.

        ENGAGING mathematical content has some or most of these characteristics:
        - Uses intuitive explanations and relatable examples
        - Provides clear language that helps deepen understanding
        - Makes abstract concepts accessible and interesting
        - Includes explanatory content that builds conceptual understanding
        - Presents reasoning that is easy to follow
        - Balances formal notation with helpful explanations

        NOT ENGAGING mathematical content typically has these characteristics:
        - Uses primarily formal mathematical notation and terminology
        - Provides minimal explanations beyond the calculations
        - Focuses exclusively on necessary calculations and logical steps
        - Lacks intuitive explanations or relatable examples
        - Uses dense, technical language that may be harder to follow
        - Prioritizes brevity and formality over accessibility

        Evaluate ONLY the engagingness of the content, NOT its correctness.

        Your evaluation must be in JSON format with two fields:
        {"analysis": "<specific reasons why the content is or is not engaging>", "judgment": "<ENGAGING or NOT ENGAGING>"}"""

        user_prompt = f"""Please evaluate the following mathematical content:

        PROBLEM:
        {problem}

        SOLUTION:
        {solution}
        """

        return system_prompt, user_prompt

    def evaluate_solution(self, problem, solution):
        """Evaluate a single math solution using vLLM"""
        # Create prompt
        system_prompt, user_prompt = self.create_judge_prompt(problem, solution)

        # Format messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate with vLLM
        outputs = self.model.generate([formatted_prompt], self.sampling_params)

        # Extract the generated text
        response = outputs[0].outputs[0].text.strip()

        # Try to extract JSON from the response first
        try:
            # Find JSON object in the response
            json_match = re.search(r"\{.*?\}", response.replace("\n", " "), re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                if "judgment" in parsed_json:
                    judgement = parsed_json["judgment"]
                    if (
                        "ENGAGING" in judgement.upper()
                        and "NOT ENGAGING" not in judgement.upper()
                    ):
                        return "engaging", response
                    elif "NOT ENGAGING" in judgement.upper():
                        return "not_engaging", response
        except Exception as e:
            # If JSON parsing fails, fall back to keyword matching
            pass

        # Fall back to keyword matching if JSON parsing fails
        if "ENGAGING" in response.upper() and "NOT ENGAGING" not in response.upper():
            judgment = "engaging"
        elif "NOT ENGAGING" in response.upper():
            judgment = "not_engaging"
        else:
            # Check for the last occurrence of engaging/not engaging
            lines = response.split("\n")
            for line in reversed(lines):
                if "ENGAGING" in line.upper() and "NOT ENGAGING" not in line.upper():
                    judgment = "engaging"
                    break
                elif "NOT ENGAGING" in line.upper():
                    judgment = "not_engaging"
                    break
            else:
                # Default if no clear judgment is found
                judgment = "unclear"

        return judgment, response


def evaluate_jsonl_engagement(input_file, output_file):
    """
    Evaluate engagement of math problems and solutions from a JSONL file
    """
    # Initialize the engagement judge
    judge = VLLMEngagementJudge()

    # Prepare results tracking
    results = []
    engagement_stats = {"total": 0, "engaging": 0, "not_engaging": 0, "unclear": 0}

    # Read the JSONL file
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # Parse the JSON object
                data = json.loads(line.strip())

                # Extract problem and solution
                problem = data.get("question", "")
                solution = data.get("model_response", "")

                # Evaluate engagement
                judgment, judge_response = judge.evaluate_solution(problem, solution)

                # Update statistics
                engagement_stats["total"] += 1
                if judgment == "engaging":
                    engagement_stats["engaging"] += 1
                elif judgment == "not_engaging":
                    engagement_stats["not_engaging"] += 1
                else:
                    engagement_stats["unclear"] += 1

                # Prepare result object
                result = {
                    "id": data.get("id", ""),
                    "problem": problem,
                    "solution": solution,
                    "engagement_judgment": judgment,
                    "judge_response": judge_response,
                }
                results.append(result)

                # Print progress
                print(
                    f"Processed {engagement_stats['total']} problems. Last judgment: {judgment}"
                )

                # Save intermediate results
                with open(output_file, "w", encoding="utf-8") as outf:
                    json.dump(
                        {"stats": engagement_stats, "results": results}, outf, indent=2
                    )

                # Clear CUDA cache periodically
                if engagement_stats["total"] % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"Error processing line: {e}")

    # Print final statistics
    print("\nFinal Engagement Statistics:")
    print(f"Total Problems: {engagement_stats['total']}")
    print(
        f"Engaging Solutions: {engagement_stats['engaging']} ({engagement_stats['engaging']/engagement_stats['total']*100:.2f}%)"
    )
    print(
        f"Not Engaging Solutions: {engagement_stats['not_engaging']} ({engagement_stats['not_engaging']/engagement_stats['total']*100:.2f}%)"
    )
    print(
        f"Unclear Judgments: {engagement_stats['unclear']} ({engagement_stats['unclear']/engagement_stats['total']*100:.2f}%)"
    )

    return results


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Evaluate mathematical solutions for engagement using vLLM"
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file containing math problems and solutions",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the output JSON file (default: input_file with '_results.json' suffix)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Model name to use for evaluation (default: Qwen/Qwen2.5-72B-Instruct)",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM (default: 4)",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization ratio (default: 0.8)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return

    # Generate output file name if not provided
    if args.output is None:
        # Remove extension and add _results.json
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_results.json"
    else:
        output_file = args.output

    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print("-" * 50)

    # Run the evaluation
    evaluate_jsonl_engagement(args.input_file, output_file)


if __name__ == "__main__":
    main()
