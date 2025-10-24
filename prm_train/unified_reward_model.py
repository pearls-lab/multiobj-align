#!/usr/bin/env python3
"""
Unified Reward Model Implementation (Binary Classification Version)
Handles PRM, Value, Preference, Engagement, and Accuracy data with binary classification objective
Optimized for accuracy metrics and binary classification head
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datasets import Dataset, DatasetDict, load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import wandb
from tqdm import tqdm
import gc
import argparse

# Environment setup
for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(k, None)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainingAccuracyCallback(TrainerCallback):
    """Log training accuracy on each eval without recursive callbacks"""

    def __init__(self):
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not hasattr(self, "trainer") or self.trainer is None:
            logger.warning("TrainingAccuracyCallback has no trainer reference attached")

    def on_evaluate(self, args, state, control, **kwargs):
        try:
            if self.trainer is None or not hasattr(self.trainer, "train_dataset"):
                return control

            ds = self.trainer.train_dataset
            n = min(1000, len(ds))
            # deterministic subset tied to seed
            rng = np.random.default_rng(42 + state.global_step)
            indices = rng.choice(len(ds), size=n, replace=False)
            train_subset = ds.select(indices.tolist())

            # predict does not trigger on_evaluate again
            pred_output = self.trainer.predict(train_subset, metric_key_prefix="train")
            metrics = pred_output.metrics  # keys like train_loss, train_accuracy, etc.

            # Log through Trainer so report_to="wandb" handles forwarding
            self.trainer.log(metrics)

        except Exception as e:
            logger.exception("Failed to log training metrics on eval")
        return control


class ClassificationDataLoader:
    """Load pre-formatted binary classification data for unified reward model training"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_formatted_data(
        self,
        prm_file: Optional[str] = None,
        preference_file: Optional[str] = None,
        value_file: Optional[str] = None,
        engagement_file: Optional[str] = None,
        accuracy_file: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        """Load formatted data from files"""
        data = {}

        if prm_file and os.path.exists(prm_file):
            self.logger.info(f"Loading PRM data from: {prm_file}")
            with open(prm_file, "r") as f:
                data["prm"] = json.load(f)
            self.logger.info(f"Loaded {len(data['prm'])} PRM examples")

        if preference_file and os.path.exists(preference_file):
            self.logger.info(f"Loading Preference data from: {preference_file}")
            with open(preference_file, "r") as f:
                data["preference"] = json.load(f)
            self.logger.info(f"Loaded {len(data['preference'])} Preference examples")

        if value_file and os.path.exists(value_file):
            self.logger.info(f"Loading Value data from: {value_file}")
            with open(value_file, "r") as f:
                data["value"] = json.load(f)
            self.logger.info(
                f"Loaded {len(data['value'])} Value examples (binary classification)"
            )

        if engagement_file and os.path.exists(engagement_file):
            self.logger.info(f"Loading Engagement data from: {engagement_file}")
            with open(engagement_file, "r") as f:
                data["engagement"] = json.load(f)
            self.logger.info(f"Loaded {len(data['engagement'])} Engagement examples")

        if accuracy_file and os.path.exists(accuracy_file):
            self.logger.info(f"Loading Accuracy data from: {accuracy_file}")
            with open(accuracy_file, "r") as f:
                data["accuracy"] = json.load(f)
            self.logger.info(f"Loaded {len(data['accuracy'])} Accuracy examples")

        return data

    def load_from_directory(self, directory: str) -> Dict[str, List[Dict]]:
        """Load all formatted data files from a directory, prioritizing classification versions"""
        self.logger.info(f"Loading formatted data from directory: {directory}")

        data = {}

        # Look for classification files first, then fall back to regular files
        file_patterns = [
            ("prm", ["formatted_prm_data_classification_", "formatted_prm_data_"]),
            (
                "preference",
                [
                    "formatted_preference_data_classification_",
                    "formatted_preference_data_",
                ],
            ),
            (
                "value",
                ["formatted_value_data_classification_", "formatted_value_data_"],
            ),
            (
                "engagement",
                [
                    "formatted_engagement_data_classification_",
                    "formatted_engagement_data_",
                ],
            ),
            (
                "accuracy",
                [
                    "formatted_accuracy_data_classification_",
                    "formatted_accuracy_data_",
                ],
            ),
        ]

        for source_type, patterns in file_patterns:
            found_file = None
            for pattern in patterns:
                for filename in os.listdir(directory):
                    if filename.startswith(pattern) and filename.endswith(".json"):
                        found_file = os.path.join(directory, filename)
                        break
                if found_file:
                    break

            if found_file:
                self.logger.info(
                    f"Loading {source_type.upper()} data from: {found_file}"
                )
                with open(found_file, "r") as f:
                    data[source_type] = json.load(f)
                self.logger.info(
                    f"Loaded {len(data[source_type])} {source_type.upper()} examples"
                )

        return data


class BinaryClassificationRewardModelTrainer:
    """Main trainer for unified reward model with optimized binary classification"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        output_dir: str = "./unified_reward_model_classification",
        max_length: int = 1024,
        prm_weight: float = 1.0,
        preference_weight: float = 1.0,
        value_weight: float = 1.0,
        engagement_weight: float = 1.0,
        accuracy_weight: float = 1.0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.weights = {
            "prm": prm_weight,
            "preference": preference_weight,
            "value": value_weight,
            "engagement": engagement_weight,
            "accuracy": accuracy_weight,
        }
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.seed = seed

        self.tokenizer = None
        self.model = None

        set_seed(seed)
        self.logger = logging.getLogger(__name__)

    def load_formatted_datasets(
        self,
        processed_data_dir: Optional[str] = None,
        prm_file: Optional[str] = None,
        preference_file: Optional[str] = None,
        value_file: Optional[str] = None,
        engagement_file: Optional[str] = None,
        accuracy_file: Optional[str] = None,
    ) -> DatasetDict:
        """Load and process formatted datasets for binary classification"""
        loader = ClassificationDataLoader()

        if processed_data_dir:
            data = loader.load_from_directory(processed_data_dir)
        else:
            data = loader.load_formatted_data(
                prm_file=prm_file,
                preference_file=preference_file,
                value_file=value_file,
                engagement_file=engagement_file,
                accuracy_file=accuracy_file,
            )

        # Combine all data sources with weights
        all_examples = []
        label_distribution = {"0": 0, "1": 0}

        for source, examples in data.items():
            weight = self.weights.get(source, 1.0)
            if weight > 0 and examples:
                # Apply weighting by replicating examples
                weighted_examples = examples * int(weight)
                all_examples.extend(weighted_examples)

                # Track label distribution
                for example in weighted_examples:
                    label = int(example["label"])
                    label_distribution[str(label)] += 1

                self.logger.info(
                    f"Added {len(weighted_examples)} {source.upper()} examples (weight: {weight})"
                )

        if not all_examples:
            raise ValueError("No data loaded! Check your file paths and data sources.")

        self.logger.info(f"Total combined examples: {len(all_examples)}")
        self.logger.info(f"Label distribution: {label_distribution}")
        self.logger.info(
            f"Class balance: {label_distribution['1']}/{len(all_examples)} positive examples ({100*label_distribution['1']/len(all_examples):.1f}%)"
        )

        # Shuffle the combined data
        random.shuffle(all_examples)

        # Validate that all labels are binary (0 or 1)
        for i, example in enumerate(all_examples):
            label = example["label"]
            if label not in [0, 1, 0.0, 1.0]:
                self.logger.warning(
                    f"Non-binary label found at index {i}: {label}. Converting to binary."
                )
                # Convert to binary if needed
                example["label"] = 1.0 if float(label) > 0.5 else 0.0

        # Create stratified splits based on labels for binary classification
        texts = [ex["text"] for ex in all_examples]
        labels = [int(ex["label"]) for ex in all_examples]  # Ensure integer labels
        sources = [ex["source"] for ex in all_examples]

        self.logger.info("Creating train/validation/test splits...")
        self.logger.info(
            f"Stratification: {'enabled' if self.stratify else 'disabled'}"
        )

        # Stratify by labels for better class balance
        stratify_by = labels if self.stratify else None

        try:
            # First split: train vs temp (val + test)
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                texts,
                labels,
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.seed,
                stratify=stratify_by,
            )

            # Second split: val vs test
            val_test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio)
            # For smaller splits, use proportional stratification if possible
            temp_stratify = (
                temp_labels
                if len(set(temp_labels)) > 1 and len(temp_labels) >= 4
                else None
            )

            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=val_test_ratio,
                random_state=self.seed,
                stratify=temp_stratify,
            )

        except ValueError as e:
            self.logger.warning(f"Stratified split failed: {e}. Using random split.")
            # Fallback to random split
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                texts,
                labels,
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.seed,
            )
            val_test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=val_test_ratio,
                random_state=self.seed,
            )

        # Log split information with class distribution
        for split_name, split_labels in [
            ("Train", train_labels),
            ("Validation", val_labels),
            ("Test", test_labels),
        ]:
            label_counts = pd.Series(split_labels).value_counts().sort_index()
            total = len(split_labels)
            self.logger.info(
                f"{split_name} set: {total} examples - {label_counts.to_dict()}"
            )
            if total > 0:
                pos_pct = (label_counts.get(1, 0) / total) * 100
                self.logger.info(f"  {split_name} positive class: {pos_pct:.1f}%")

        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

        # Create DatasetDict
        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        return dataset_dict

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for binary classification"""
        self.logger.info(f"Loading tokenizer and model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        self.tokenizer.padding_side = "right"

        # Load model config and set for binary classification
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        config.num_labels = 2  # Binary classification: 2 classes (0, 1)
        config.problem_type = "single_label_classification"

        # Ensure proper classification head initialization
        config.classifier_dropout = 0.1  # Add dropout for regularization
        config.pad_token_id = self.tokenizer.pad_token_id  # important

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # keep model and generation configs in sync
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if (
            hasattr(self.model, "generation_config")
            and self.model.generation_config is not None
        ):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        # gradient checkpointing already forces use_cache False, this is safe to set too
        self.model.config.use_cache = False

        self.logger.info("Model and tokenizer setup complete")
        self.logger.info(
            f"Model configuration: {config.num_labels} classes, {config.problem_type}"
        )
        self.logger.info(
            f"Classification head dropout: {getattr(config, 'classifier_dropout', 'default')}"
        )

    def tokenize_function(self, examples):
        """Tokenize text and ensure integer labels for classification"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        # Ensure labels are integers (0 or 1) for classification
        tokenized["labels"] = [int(label) for label in examples["label"]]
        return tokenized

    def compute_metrics(self, eval_pred):
        """Comprehensive binary classification metrics"""
        predictions, labels = eval_pred

        # For binary classification, predictions are logits of shape [batch_size, 2]
        # We take the softmax and use the probability of class 1
        probs = F.softmax(torch.from_numpy(predictions), dim=-1)[:, 1].numpy()
        predicted_classes = (probs > 0.5).astype(int)

        labels = labels.astype(int)

        # Calculate comprehensive metrics
        accuracy = accuracy_score(labels, predicted_classes)
        precision = precision_score(
            labels, predicted_classes, average="binary", zero_division=0
        )
        recall = recall_score(
            labels, predicted_classes, average="binary", zero_division=0
        )
        f1 = f1_score(labels, predicted_classes, average="binary", zero_division=0)

        # ROC-AUC (using probabilities)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0  # In case of single class in labels

        # Confusion matrix
        cm = confusion_matrix(labels, predicted_classes)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "specificity": specificity,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    def train(
        self,
        dataset: DatasetDict,
        learning_rate: float = 1e-5,
        num_epochs: int = 2,
        batch_size: int = 16,
    ):
        """Train the unified reward model with binary classification"""
        self.logger.info(
            "Starting unified reward model training (binary classification)..."
        )

        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # Calculate dynamic training schedule
        total_steps = (len(tokenized_datasets["train"]) // batch_size) * num_epochs
        calculated_warmup_steps = int(0.1 * total_steps)
        calculated_eval_steps = max(200, int(0.15 * total_steps))

        self.logger.info(f"Training Schedule:")
        self.logger.info(f"  Total steps: {total_steps}")
        self.logger.info(f"  Warmup steps: {calculated_warmup_steps} (10%)")
        self.logger.info(
            f"  Eval steps: {calculated_eval_steps} (~{total_steps//calculated_eval_steps} evaluations)"
        )
        self.logger.info(f"  Logging steps: {50}")

        # Training arguments optimized for binary classification
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=calculated_warmup_steps,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=50,  # Frequent logging for monitoring
            eval_strategy="steps",
            eval_steps=calculated_eval_steps,
            save_strategy="steps",
            save_steps=calculated_eval_steps,  # Align with eval_steps
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # Use F1 score for best model selection
            greater_is_better=True,
            report_to="wandb",
            run_name=f"unified_rm_classification_{self.model_name.split('/')[-1]}",
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            remove_unused_columns=False,
            # Additional classification-specific settings
            label_smoothing_factor=0.0,  # No label smoothing for binary classification
            seed=self.seed,
        )

        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True, return_tensors="pt"
        )

        # Create training accuracy callback
        training_callback = TrainingAccuracyCallback()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[training_callback],
        )

        # give the callback a handle to the trainer
        training_callback.trainer = trainer

        # Train
        self.logger.info("Starting training...")
        trainer.train()

        # Evaluate on all splits
        self.logger.info("Evaluating on all splits...")

        train_results = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
        val_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

        self.logger.info(f"Train results: {train_results}")
        self.logger.info(f"Validation results: {val_results}")
        self.logger.info(f"Test results: {test_results}")

        # Save final model
        self.logger.info("Saving final model...")
        trainer.save_model(f"{self.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final_model")

        # Save comprehensive training summary
        summary = {
            "model_name": self.model_name,
            "training_args": training_args.to_dict(),
            "classification_type": "binary",
            "results": {
                "train": train_results,
                "validation": val_results,
                "test": test_results,
            },
            "data_weights": self.weights,
            "dataset_sizes": {
                "train": len(tokenized_datasets["train"]),
                "validation": len(tokenized_datasets["validation"]),
                "test": len(tokenized_datasets["test"]),
            },
            "model_config": {
                "num_labels": 2,
                "problem_type": "single_label_classification",
                "max_length": self.max_length,
            },
        }

        with open(f"{self.output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Training completed!")
        self.logger.info(
            f"Final Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}"
        )
        self.logger.info(f"Final Test F1: {test_results.get('eval_f1', 0):.4f}")

        return trainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified Reward Model Trainer (Binary Classification)"
    )

    # Data loading arguments
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        help="Directory containing processed data files",
    )
    parser.add_argument("--prm_file", type=str, help="Path to formatted PRM data file")
    parser.add_argument(
        "--preference_file", type=str, help="Path to formatted preference data file"
    )
    parser.add_argument(
        "--value_file", type=str, help="Path to formatted value data file"
    )
    parser.add_argument(
        "--engagement_file", type=str, help="Path to formatted engagement data file"
    )
    parser.add_argument(
        "--accuracy_file", type=str, help="Path to formatted accuracy data file"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Pre-trained model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./unified_reward_model_classification",
        help="Output directory for model and logs",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation steps (auto-calculated if None)",
    )
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")

    # Data weighting arguments
    parser.add_argument(
        "--prm_weight", type=float, default=1.0, help="Weight for PRM data"
    )
    parser.add_argument(
        "--preference_weight",
        type=float,
        default=1.0,
        help="Weight for preference data",
    )
    parser.add_argument(
        "--value_weight", type=float, default=1.0, help="Weight for value data"
    )
    parser.add_argument(
        "--engagement_weight",
        type=float,
        default=1.0,
        help="Weight for engagement data",
    )
    parser.add_argument(
        "--accuracy_weight",
        type=float,
        default=1.0,
        help="Weight for accuracy data",
    )

    # Split arguments
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training data ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation data ratio"
    )
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test data ratio")
    parser.add_argument(
        "--no_stratify", action="store_true", help="Disable stratified splitting"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Validate arguments
    if not args.processed_data_dir and not any(
        [
            args.prm_file,
            args.preference_file,
            args.value_file,
            args.engagement_file,
            args.accuracy_file,
        ]
    ):
        raise ValueError(
            "Either --processed_data_dir or individual data files must be specified"
        )

    # Initialize trainer
    trainer = BinaryClassificationRewardModelTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        prm_weight=args.prm_weight,
        preference_weight=args.preference_weight,
        value_weight=args.value_weight,
        engagement_weight=args.engagement_weight,
        accuracy_weight=args.accuracy_weight,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=not args.no_stratify,
        seed=args.seed,
    )

    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()

    # Load datasets
    dataset = trainer.load_formatted_datasets(
        processed_data_dir=args.processed_data_dir,
        prm_file=args.prm_file,
        preference_file=args.preference_file,
        value_file=args.value_file,
        engagement_file=args.engagement_file,
        accuracy_file=args.accuracy_file,
    )

    # Initialize wandb
    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(
        project="unified_reward_model_classification",
        name=f"unified_rm_classification_{args.model_name.split('/')[-1]}",
        config=vars(args),
    )

    # Train
    trainer.train(
        dataset,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    # Cleanup
    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
