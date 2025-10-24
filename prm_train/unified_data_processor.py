#!/usr/bin/env python3
"""
Unified Data Processor: Sample, Format, and Process Multiple Data Sources.
This script combines sampling and formatting functionality into a single pipeline.
"""

import os
import json
import argparse
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDataProcessor:
    """Process all data types into unified format with proper User/Assistant spacing"""

    def process_prm_data(self, prm_data: List[Dict]) -> List[Dict]:
        """Process PRM data with incremental step accumulation from conversation format"""
        unified_examples = []

        for example in prm_data:
            # Handle new conversation format (check both 'conversation' and 'conversations')
            if "conversation" in example:
                conversation = example["conversation"]
            elif "conversations" in example:
                conversation = example["conversations"]
            else:
                conversation = None

            if conversation:
                # Extract question (first user message) and steps (subsequent user messages)
                question = None
                steps = []
                labels = []

                # Process messages in pairs (user step + assistant label)
                i = 0
                while i < len(conversation):
                    if conversation[i]["role"] == "user":
                        if question is None:
                            # First user message is the question
                            question = conversation[i]["content"]
                        else:
                            # Subsequent user messages are steps
                            step_content = conversation[i]["content"]
                            steps.append(step_content)

                            # Look for the corresponding assistant label
                            if (
                                i + 1 < len(conversation)
                                and conversation[i + 1]["role"] == "assistant"
                            ):
                                label_content = conversation[i + 1]["content"].strip()
                                if label_content == "+":
                                    labels.append(1.0)
                                elif label_content == "-":
                                    labels.append(0.0)
                                else:
                                    # Invalid label format - this is an error
                                    logger.error(
                                        f"Invalid assistant label '{label_content}', expected '+' or '-'. Skipping conversation."
                                    )
                                    # Set flag to skip this conversation
                                    steps = []
                                    labels = []
                                    break
                                i += 1  # Skip the assistant message since we processed it
                            else:
                                # No corresponding assistant label - this is an error
                                logger.error(
                                    f"Missing assistant label for step '{step_content[:50]}...'. Skipping conversation."
                                )
                                # Set flag to skip this conversation
                                steps = []
                                labels = []
                                break
                    i += 1

                if not question or not steps:
                    logger.warning(
                        "Skipping invalid conversation: missing question or steps"
                    )
                    continue

                # Ensure steps and labels have the same length
                if len(steps) != len(labels):
                    logger.warning(
                        f"Mismatch between steps ({len(steps)}) and labels ({len(labels)}), truncating to shorter length"
                    )
                    min_len = min(len(steps), len(labels))
                    steps = steps[:min_len]
                    labels = labels[:min_len]

            else:
                # Handle old format: question/steps/step_labels (for backward compatibility)
                question = example["question"]
                steps = example["steps"]
                step_labels = example["step_labels"]

                # Convert step_labels to binary labels
                labels = [1.0 if label == "+" else 0.0 for label in step_labels]

            # Generate incremental examples with proper spacing
            accumulated_text = f"User: {question}\n\nAssistant: "

            for i, (step, label) in enumerate(zip(steps, labels)):
                if i == 0:
                    accumulated_text += step
                else:
                    accumulated_text += f"\n\n{step}"

                unified_examples.append(
                    {"text": accumulated_text, "label": label, "source": "prm"}
                )

        logger.info(
            f"Processed {len(unified_examples)} PRM examples from {len(prm_data)} original samples"
        )
        return unified_examples

    def process_value_data(self, value_data: List[Dict]) -> List[Dict]:
        """Process value data with binary classification (>0.85 = 1, <=0.85 = 0)"""
        unified_examples = []

        for example in value_data:
            question = example["question"]
            steps = example["response_steps"]
            values = example["step_value"]

            # Find max value for this example for normalization
            max_value = max(values) if values else 1.0

            # Generate incremental examples with proper spacing
            accumulated_text = f"User: {question}\n\nAssistant: "

            for i, (step, value) in enumerate(zip(steps, values)):
                if i == 0:
                    accumulated_text += step
                else:
                    accumulated_text += f"\n\n{step}"

                # Normalize to [0,1] range first
                normalized_value = value / max_value if max_value > 0 else 0.0

                # Convert to binary classification: >0.85 = 1, <=0.85 = 0
                binary_label = 1.0 if normalized_value > 0.85 else 0.0

                unified_examples.append(
                    {
                        "text": accumulated_text,
                        "label": binary_label,
                        "source": "value",
                    }
                )

        logger.info(
            f"Processed {len(unified_examples)} Value examples from {len(value_data)} original samples"
        )
        logger.info(
            f"Value data converted to binary classification (>0.85 normalized = 1, <=0.85 = 0)"
        )
        return unified_examples

    def process_preference_data(self, preference_data: List[Dict]) -> List[Dict]:
        """Process preference data with chosen=1.0, rejected=0.0"""
        unified_examples = []

        for example in preference_data:
            chosen = example["chosen"]
            rejected = example["rejected"]

            chosen_text = self.format_conversation(chosen)
            rejected_text = self.format_conversation(rejected)

            unified_examples.extend(
                [
                    {"text": chosen_text, "label": 1.0, "source": "preference"},
                    {"text": rejected_text, "label": 0.0, "source": "preference"},
                ]
            )

        logger.info(
            f"Processed {len(unified_examples)} Preference examples from {len(preference_data)} original pairs"
        )
        return unified_examples

    def process_engagement_data(self, engagement_data: List[Dict]) -> List[Dict]:
        """Process engagement data with binary labels converted to float (matches engage_train2.py format)
        Filters out samples with ≤1 Assistant turn to ensure multi-turn conversations only.
        """
        unified_examples = []
        filtered_count = 0

        for example in engagement_data:
            if "prompt_answer" in example:
                # Handle list format - directly join without role modifications (matches engage_train2.py)
                conversation = example["prompt_answer"]
                text = "\n\n".join(conversation)
            else:
                # Handle message format
                text = self.format_conversation(example.get("messages", []))

            # Count Assistant turns - exclude samples with only 1 Assistant turn
            assistant_count = text.count("Assistant:")
            if assistant_count <= 1:
                filtered_count += 1
                continue

            label = float(example["label"])  # Convert to float (0.0 or 1.0)

            unified_examples.append(
                {"text": text, "label": label, "source": "engagement"}
            )

        logger.info(
            f"Processed {len(unified_examples)} Engagement examples from {len(engagement_data)} original samples"
        )
        logger.info(
            f"Filtered out {filtered_count} samples with ≤1 Assistant turn ({filtered_count/len(engagement_data)*100:.1f}%)"
        )
        return unified_examples

    def process_accuracy_data(self, accuracy_data: List[Dict]) -> List[Dict]:
        """Process accuracy data with binary labels converted to float (excludes last conversation element)
        Filters out samples with ≤1 Assistant turn to ensure multi-turn conversations only.
        """
        unified_examples = []
        filtered_count = 0

        for example in accuracy_data:
            if "prompt_answer" in example:
                # Handle list format - join all but the last element
                conversation = example["prompt_answer"]
                text = "\n\n".join(conversation[:-1])
            else:
                # Handle message format
                text = self.format_conversation(example.get("messages", []))

            # Count Assistant turns - exclude samples with only 1 Assistant turn
            assistant_count = text.count("Assistant:")
            if assistant_count <= 1:
                filtered_count += 1
                continue

            label = float(example["label"])  # Convert to float (0.0 or 1.0)

            unified_examples.append(
                {"text": text, "label": label, "source": "accuracy"}
            )

        logger.info(
            f"Processed {len(unified_examples)} Accuracy examples from {len(accuracy_data)} original samples"
        )
        logger.info(
            f"Filtered out {filtered_count} samples with ≤1 Assistant turn ({filtered_count/len(accuracy_data)*100:.1f}%)"
        )
        return unified_examples

    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Convert message list to unified User/Assistant format with proper spacing"""
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        return "\n\n".join(lines)


class UnifiedDataProcessorPipeline:
    """Complete pipeline for sampling, formatting, and processing multiple data sources"""

    def __init__(self, output_dir: str = "./processed_data"):
        self.output_dir = output_dir
        self.processor = UnifiedDataProcessor()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamp for this processing run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Default data source configuration
        self.default_config = {
            "prm_data": {
                "source": "Jennny/strict_mc_label_problem_handle",
                "sample_size": 4000,
                "description": "Process Reward Model data for step-by-step reasoning",
            },
            "preference_data": {
                "source": "Jennny/cleaned_ultrafeedback_preferences",
                "sample_size": 8000,
                "description": "Human preference data for response quality",
            },
            "value_data": {
                "sources": ["final_mc_data_09.json", "final_mc_data2_09.json"],
                "sample_size": 4500,  # Per file
                "description": "Value model data for solution progress estimation (converted to binary)",
            },
            "engagement_data": {
                "source": "Jennny/final_eng_rate_latest2",
                "sample_size": 2000,
                "description": "Engagement scoring data for conversation quality (multi-turn only)",
            },
            "accuracy_data": {
                "source": "Jennny/final_acc_rate_latest3",
                "sample_size": 2000,
                "description": "Accuracy scoring data for conversation quality (excludes last turn, multi-turn only)",
            },
        }

    def sample_prm_data(self, source: str, sample_size: int) -> List[Dict]:
        """Sample PRM data from HuggingFace dataset"""
        if sample_size <= 0:
            logger.info("PRM sampling disabled (sample_size=0)")
            return []

        logger.info(f"Sampling {sample_size} PRM examples from {source}...")

        try:
            dataset = load_dataset(source)
            full_data = list(dataset["train"])

            # Sample randomly
            actual_sample_size = min(sample_size, len(full_data))
            sampled_data = random.sample(full_data, actual_sample_size)

            logger.info(
                f"Sampled {len(sampled_data)} PRM examples from {len(full_data)} total"
            )
            return sampled_data

        except Exception as e:
            logger.error(f"ERROR: Error sampling PRM data from {source}: {e}")
            return []

    def sample_preference_data(self, source: str, sample_size: int) -> List[Dict]:
        """Sample preference data from HuggingFace dataset"""
        if sample_size <= 0:
            logger.info("Preference sampling disabled (sample_size=0)")
            return []

        logger.info(f"Sampling {sample_size} preference examples from {source}...")

        try:
            dataset = load_dataset(source)

            # Handle different splits
            if "train_prefs" in dataset:
                full_data = list(dataset["train_prefs"])
            elif "train" in dataset:
                full_data = list(dataset["train"])
            else:
                # Try to find any available split
                available_splits = list(dataset.keys())
                if available_splits:
                    split_name = available_splits[0]
                    full_data = list(dataset[split_name])
                    logger.info(
                        f"Using split '{split_name}' (available: {available_splits})"
                    )
                else:
                    raise ValueError(f"No usable splits found in dataset {source}")

            # Sample randomly
            actual_sample_size = min(sample_size, len(full_data))
            sampled_data = random.sample(full_data, actual_sample_size)

            logger.info(
                f"Sampled {len(sampled_data)} preference examples from {len(full_data)} total"
            )
            return sampled_data

        except Exception as e:
            logger.error(f"ERROR: Error sampling preference data from {source}: {e}")
            return []

    def sample_value_data(self, sources: List[str], sample_size: int) -> List[Dict]:
        """Sample value data from local JSON files"""
        if sample_size <= 0:
            logger.info("Value sampling disabled (sample_size=0)")
            return []

        logger.info(f"Sampling {sample_size} value examples from each file...")

        all_sampled_data = []

        for file_path in sources:
            try:
                logger.info(f"Processing {file_path}...")

                if not os.path.exists(file_path):
                    logger.warning(f"ERROR: File not found: {file_path}, skipping...")
                    continue

                with open(file_path, "r") as f:
                    full_data = json.load(f)

                # Sample randomly
                actual_sample_size = min(sample_size, len(full_data))
                sampled_data = random.sample(full_data, actual_sample_size)
                all_sampled_data.extend(sampled_data)

                logger.info(f"Sampled {len(sampled_data)} examples from {file_path}")

            except Exception as e:
                logger.error(f"ERROR: Error sampling from {file_path}: {e}")

        logger.info(f"Total sampled value examples: {len(all_sampled_data)}")
        return all_sampled_data

    def sample_engagement_data(self, source: str, sample_size: int) -> List[Dict]:
        """Sample engagement data from HuggingFace dataset"""
        if sample_size <= 0:
            logger.info("Engagement sampling disabled (sample_size=0)")
            return []

        logger.info(f"Sampling {sample_size} engagement examples from {source}...")

        try:
            dataset = load_dataset(source)
            full_data = list(dataset["train"])

            # Sample randomly
            actual_sample_size = min(sample_size, len(full_data))
            sampled_data = random.sample(full_data, actual_sample_size)

            logger.info(
                f"Sampled {len(sampled_data)} engagement examples from {len(full_data)} total"
            )
            return sampled_data

        except Exception as e:
            logger.error(f"ERROR: Error sampling engagement data from {source}: {e}")
            return []

    def sample_accuracy_data(self, source: str, sample_size: int) -> List[Dict]:
        """Sample accuracy data from HuggingFace dataset"""
        if sample_size <= 0:
            logger.info("Accuracy sampling disabled (sample_size=0)")
            return []

        logger.info(f"Sampling {sample_size} accuracy examples from {source}...")

        try:
            dataset = load_dataset(source)
            full_data = list(dataset["train"])

            # Sample randomly
            actual_sample_size = min(sample_size, len(full_data))
            sampled_data = random.sample(full_data, actual_sample_size)

            logger.info(
                f"Sampled {len(sampled_data)} accuracy examples from {len(full_data)} total"
            )
            return sampled_data

        except Exception as e:
            logger.error(f"ERROR: Error sampling accuracy data from {source}: {e}")
            return []

    def process_and_save_data_type(
        self, data_type: str, raw_data: List[Dict], process_func
    ) -> Optional[str]:
        """Process and save a specific data type"""
        if not raw_data:
            logger.info(f"No {data_type} data to process, skipping...")
            return None

        logger.info(f"Processing {len(raw_data)} {data_type} examples...")

        # Process the data
        processed_examples = process_func(raw_data)

        if not processed_examples:
            logger.warning(f"ERROR: No processed examples generated for {data_type}")
            return None

        # Save processed data with classification suffix
        output_filename = (
            f"formatted_{data_type}_data_classification_{self.timestamp}.json"
        )
        output_path = os.path.join(self.output_dir, output_filename)

        with open(output_path, "w") as f:
            json.dump(processed_examples, f, indent=2)

        logger.info(
            f"Saved {len(processed_examples)} formatted {data_type} examples to: {output_path}"
        )

        # Save metadata
        metadata = {
            "data_type": data_type,
            "classification_mode": True,
            "num_raw_examples": len(raw_data),
            "num_processed_examples": len(processed_examples),
            "processing_timestamp": self.timestamp,
            "output_file": output_path,
            "expansion_ratio": (
                len(processed_examples) / len(raw_data) if raw_data else 0
            ),
        }

        # Add specific classification info for value data
        if data_type == "value":
            metadata["value_classification_info"] = (
                "Labels converted to binary: >0.85 normalized = 1, <=0.85 = 0"
            )

        metadata_path = os.path.join(
            self.output_dir,
            f"{data_type}_metadata_classification_{self.timestamp}.json",
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    def run_complete_pipeline(
        self,
        # Sample sizes
        prm_sample_size: int = 2000,
        preference_sample_size: int = 5000,
        value_sample_size: int = 1500,
        engagement_sample_size: int = 2000,
        accuracy_sample_size: int = 2000,
        # Data sources
        prm_source: str = None,
        preference_source: str = None,
        value_sources: List[str] = None,
        engagement_source: str = None,
        accuracy_source: str = None,
    ) -> Dict[str, str]:
        """Run the complete sampling and processing pipeline"""

        logger.info("=" * 80)
        logger.info("STARTING UNIFIED DATA PROCESSING PIPELINE (CLASSIFICATION)")
        logger.info("=" * 80)

        # Use defaults if not specified
        prm_source = prm_source or self.default_config["prm_data"]["source"]
        preference_source = (
            preference_source or self.default_config["preference_data"]["source"]
        )
        value_sources = value_sources or self.default_config["value_data"]["sources"]
        engagement_source = (
            engagement_source or self.default_config["engagement_data"]["source"]
        )
        accuracy_source = (
            accuracy_source or self.default_config["accuracy_data"]["source"]
        )

        # Log configuration
        logger.info("PROCESSING CONFIGURATION:")
        logger.info(f"  Output Directory: {self.output_dir}")
        logger.info(f"  Timestamp: {self.timestamp}")
        logger.info(f"  Classification Mode: ENABLED (Value data converted to binary)")
        logger.info("")
        logger.info("SAMPLE SIZES:")
        logger.info(f"  PRM: {prm_sample_size:,} conversations")
        logger.info(f"  Preference: {preference_sample_size:,} pairs")
        logger.info(f"  Value: {value_sample_size:,} per file (→ binary labels)")
        logger.info(
            f"  Engagement: {engagement_sample_size:,} examples (multi-turn only)"
        )
        logger.info(
            f"  Accuracy: {accuracy_sample_size:,} examples (excludes last turn, multi-turn only)"
        )
        logger.info("")
        logger.info("DATA SOURCES:")
        logger.info(f"  PRM: {prm_source}")
        logger.info(f"  Preference: {preference_source}")
        logger.info(f"  Value: {value_sources}")
        logger.info(f"  Engagement: {engagement_source}")
        logger.info(f"  Accuracy: {accuracy_source}")
        logger.info("=" * 80)

        saved_files = {}
        processing_stats = {}

        # 1. Sample and process PRM data
        logger.info("\nSTEP 1: PRM DATA")
        logger.info("-" * 40)
        prm_data = self.sample_prm_data(prm_source, prm_sample_size)
        if prm_data:
            prm_path = self.process_and_save_data_type(
                "prm", prm_data, self.processor.process_prm_data
            )
            if prm_path:
                saved_files["prm"] = prm_path
                processing_stats["prm"] = {
                    "raw_samples": len(prm_data),
                    "processed_examples": "See metadata file",
                }

        # 2. Sample and process Preference data
        logger.info("\nSTEP 2: PREFERENCE DATA")
        logger.info("-" * 40)
        pref_data = self.sample_preference_data(
            preference_source, preference_sample_size
        )
        if pref_data:
            pref_path = self.process_and_save_data_type(
                "preference", pref_data, self.processor.process_preference_data
            )
            if pref_path:
                saved_files["preference"] = pref_path
                processing_stats["preference"] = {
                    "raw_samples": len(pref_data),
                    "processed_examples": "See metadata file",
                }

        # 3. Sample and process Value data (with binary classification)
        logger.info("\nSTEP 3: VALUE DATA (BINARY CLASSIFICATION)")
        logger.info("-" * 40)
        value_data = self.sample_value_data(value_sources, value_sample_size)
        if value_data:
            value_path = self.process_and_save_data_type(
                "value", value_data, self.processor.process_value_data
            )
            if value_path:
                saved_files["value"] = value_path
                processing_stats["value"] = {
                    "raw_samples": len(value_data),
                    "processed_examples": "See metadata file",
                    "classification_note": "Labels converted to binary (>0.85 = 1, <=0.85 = 0)",
                }

        # 4. Sample and process Engagement data
        logger.info("\nSTEP 4: ENGAGEMENT DATA (MULTI-TURN ONLY)")
        logger.info("-" * 40)
        engagement_data = self.sample_engagement_data(
            engagement_source, engagement_sample_size
        )
        if engagement_data:
            engagement_path = self.process_and_save_data_type(
                "engagement", engagement_data, self.processor.process_engagement_data
            )
            if engagement_path:
                saved_files["engagement"] = engagement_path
                processing_stats["engagement"] = {
                    "raw_samples": len(engagement_data),
                    "processed_examples": "See metadata file",
                }

        # 5. Sample and process Accuracy data
        logger.info("\nSTEP 5: ACCURACY DATA (EXCLUDES LAST TURN, MULTI-TURN ONLY)")
        logger.info("-" * 40)
        accuracy_data = self.sample_accuracy_data(accuracy_source, accuracy_sample_size)
        if accuracy_data:
            accuracy_path = self.process_and_save_data_type(
                "accuracy", accuracy_data, self.processor.process_accuracy_data
            )
            if accuracy_path:
                saved_files["accuracy"] = accuracy_path
                processing_stats["accuracy"] = {
                    "raw_samples": len(accuracy_data),
                    "processed_examples": "See metadata file",
                }

        # 6. Create comprehensive summary
        logger.info("\nSTEP 6: CREATING SUMMARY")
        logger.info("-" * 40)

        summary = {
            "processing_timestamp": self.timestamp,
            "output_directory": self.output_dir,
            "classification_mode": True,
            "configuration": {
                "sample_sizes": {
                    "prm": prm_sample_size,
                    "preference": preference_sample_size,
                    "value": value_sample_size,
                    "engagement": engagement_sample_size,
                    "accuracy": accuracy_sample_size,
                },
                "data_sources": {
                    "prm": prm_source,
                    "preference": preference_source,
                    "value": value_sources,
                    "engagement": engagement_source,
                    "accuracy": accuracy_source,
                },
            },
            "saved_files": saved_files,
            "processing_stats": processing_stats,
            "classification_info": {
                "value_data": "Labels converted to binary classification (>0.85 normalized = 1, <=0.85 = 0)",
                "other_data": "Already binary (prm, preference, engagement, accuracy)",
                "conversation_filtering": "Engagement and accuracy data filtered to multi-turn conversations only (>1 Assistant turn)",
            },
            "next_steps": {
                "train_command": self._generate_training_command(saved_files),
                "formatted_files_ready": len(saved_files) > 0,
            },
        }

        summary_path = os.path.join(
            self.output_dir, f"processing_summary_classification_{self.timestamp}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 6. Final report
        logger.info("=" * 80)
        logger.info("UNIFIED DATA PROCESSING COMPLETE (CLASSIFICATION)")
        logger.info("=" * 80)
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Summary File: {summary_path}")
        logger.info(f"Classification Mode: ENABLED")
        logger.info("")
        logger.info("PROCESSED DATA FILES:")

        total_datasets = 0
        for data_type, file_path in saved_files.items():
            if file_path:
                total_datasets += 1
                logger.info(f"  {data_type.upper()}: {os.path.basename(file_path)}")

        if total_datasets == 0:
            logger.warning("ERROR: No data was successfully processed!")
        else:
            logger.info(f"\nSuccessfully processed {total_datasets} data types")

        logger.info("")
        logger.info("NEXT STEPS:")
        if saved_files:
            logger.info("1. Use the generated training command below:")
            logger.info("")
            logger.info(summary["next_steps"]["train_command"])
            logger.info("")
            logger.info("2. Or use individual files for custom training")
        else:
            logger.info("ERROR: No data processed. Check error messages above.")

        logger.info("=" * 80)

        return saved_files

    def _generate_training_command(self, saved_files: Dict[str, str]) -> str:
        """Generate the training command for the processed data"""
        if not saved_files:
            return "# No processed files available"

        command_parts = ["python unified_reward_model_classification.py"]

        command_parts.append(f"--processed_data_dir {self.output_dir}")

        command_parts.extend(
            [
                "--model_name meta-llama/Llama-3.1-8B",
                "--output_dir ./unified_model_classification",
                "--batch_size 16",
                "--learning_rate 1e-5",
                "--num_epochs 2",
            ]
        )

        return " \\\n  ".join(command_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Data Processor: Sample, Format, and Process Multiple Data Sources (Classification Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default configuration (with value data converted to binary)
  python unified_data_processor_classification.py
  
  # Mathematical focus (emphasize PRM and Value data)
  python unified_data_processor_classification.py \\
    --prm_sample_size 4000 \\
    --value_sample_size 3000 \\
    --preference_sample_size 1000 \\
    --engagement_sample_size 0 \\
    --accuracy_sample_size 0
    
  # Conversational focus (emphasize Preference, Engagement, and Accuracy)
  python unified_data_processor_classification.py \\
    --prm_sample_size 1000 \\
    --value_sample_size 500 \\
    --preference_sample_size 8000 \\
    --engagement_sample_size 4000 \\
    --accuracy_sample_size 4000
        """,
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        default="./processed_data_classification",
        help="Directory to save processed data (default: ./processed_data_classification)",
    )

    # Sample sizes
    parser.add_argument(
        "--prm_sample_size",
        type=int,
        default=2000,
        help="Number of PRM conversations to sample (default: 2000, 0=disable)",
    )
    parser.add_argument(
        "--preference_sample_size",
        type=int,
        default=5000,
        help="Number of preference pairs to sample (default: 5000, 0=disable)",
    )
    parser.add_argument(
        "--value_sample_size",
        type=int,
        default=1500,
        help="Number of value examples per file to sample (default: 1500, 0=disable) - converted to binary",
    )
    parser.add_argument(
        "--engagement_sample_size",
        type=int,
        default=2000,
        help="Number of engagement examples to sample (default: 2000, 0=disable)",
    )
    parser.add_argument(
        "--accuracy_sample_size",
        type=int,
        default=2000,
        help="Number of accuracy examples to sample (default: 2000, 0=disable)",
    )

    # Data sources
    parser.add_argument(
        "--prm_source",
        default="Jennny/strict_mc_label_problem_handle",
        help="HuggingFace dataset for PRM data",
    )
    parser.add_argument(
        "--preference_source",
        default="Jennny/cleaned_ultrafeedback_preferences",
        help="HuggingFace dataset for preference data",
    )
    parser.add_argument(
        "--value_sources",
        nargs="+",
        default=["final_mc_data_09.json", "final_mc_data2_09.json"],
        help="Local JSON files for value data (will be converted to binary classification)",
    )
    parser.add_argument(
        "--engagement_source",
        default="Jennny/final_eng_rate_latest2",
        help="HuggingFace dataset for engagement data",
    )
    parser.add_argument(
        "--accuracy_source",
        default="Jennny/final_acc_rate_latest3",
        help="HuggingFace dataset for accuracy data",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Create processor
    processor = UnifiedDataProcessorPipeline(output_dir=args.output_dir)

    # Run the complete pipeline
    saved_files = processor.run_complete_pipeline(
        # Sample sizes
        prm_sample_size=args.prm_sample_size,
        preference_sample_size=args.preference_sample_size,
        value_sample_size=args.value_sample_size,
        engagement_sample_size=args.engagement_sample_size,
        accuracy_sample_size=args.accuracy_sample_size,
        # Data sources
        prm_source=args.prm_source,
        preference_source=args.preference_source,
        value_sources=args.value_sources,
        engagement_source=args.engagement_source,
        accuracy_source=args.accuracy_source,
    )

    # Exit with appropriate code
    if saved_files:
        logger.info("Pipeline completed successfully!")
        exit(0)
    else:
        logger.error("ERROR: Pipeline failed - no data was processed")
        exit(1)


if __name__ == "__main__":
    main()
