#!/usr/bin/env python3
"""
Reward Model Training Script for Assistant Reply Reranking
Based on accuracy dataset with binary labels (0 and 1)
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datasets import Dataset, DatasetDict, load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import wandb
from tqdm import tqdm
import gc

# Environment variables for distributed training control
for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(k, None)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


class RewardModelTrainer:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        max_length: int = 2048,
        learning_rate: float = 5e-6,  # Much lower learning rate
        batch_size: int = 32,
        num_epochs: int = 3,  # Reduce epochs to prevent overfitting
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        output_dir: str = "./reward_model_output",
        val_ratio: float = 0.05,  # Validation set ratio
        test_ratio: float = 0.05,  # Test set ratio
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Initialize tokenizer with proper padding configuration
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Properly configure padding token for LLaMA models
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # Fallback: add a new padding token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Set padding side - right padding is typically better for classification
        self.tokenizer.padding_side = "right"

        # Initialize model with AutoModelForSequenceClassification
        logger.info(f"Loading model {model_name} with device_map='auto'")

        # Configure model for binary classification (num_labels=2 for labels 0 and 1)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = 2  # Binary classification: labels 0 and 1
        config.problem_type = "single_label_classification"
        # Set pad_token_id in config to match tokenizer
        config.pad_token_id = self.tokenizer.pad_token_id

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            device_map="auto",
            trust_remote_code=True,
        )

        # Resize token embeddings if we added new tokens
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Count parameters (no layer freezing needed with AutoModelForSequenceClassification)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logger.info(f"Initialized reward model trainer with {self.model_name}")
        logger.info(f"Model has {total_params:,} parameters")
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})"
        )
        logger.info(
            f"Model configured for binary classification with {config.num_labels} labels"
        )

    def load_and_process_dataset(self, dataset_name: str) -> DatasetDict:
        """Load and process the accuracy dataset, filtering out samples with ≤1 Assistant turn"""
        logger.info(f"Loading dataset: {dataset_name}")

        # Load dataset from HuggingFace
        raw_dataset = load_dataset(dataset_name)

        # Process the dataset
        processed_data = []
        filtered_count = 0

        # Handle the case where the dataset has only one split called "train"
        if len(raw_dataset) == 1 and "train" in raw_dataset:
            split_data = raw_dataset["train"]
            logger.info(f"Processing single split with {len(split_data)} examples")

            for example in tqdm(split_data, desc="Processing examples"):
                # Parse the conversation format
                conversation = example["prompt_answer"]
                label = int(example["label"])

                # Convert to Assistant/Student role-tagged text
                text = self.format_conversation(conversation)

                # Count Assistant turns - exclude samples with only 1 Assistant turn
                assistant_count = text.count("Assistant:")
                if assistant_count <= 1:
                    filtered_count += 1
                    continue

                processed_data.append(
                    {"text": text, "label": label, "conversation": conversation}
                )
        else:
            # Handle multiple splits
            for split_name, split_data in raw_dataset.items():
                logger.info(
                    f"Processing {split_name} split with {len(split_data)} examples"
                )

                for example in tqdm(split_data, desc=f"Processing {split_name}"):
                    # Parse the conversation format
                    conversation = example["prompt_answer"]
                    label = int(example["label"])

                    # Convert to Assistant/Student role-tagged text
                    text = self.format_conversation(conversation)

                    # Count Assistant turns - exclude samples with only 1 Assistant turn
                    assistant_count = text.count("Assistant:")
                    if assistant_count <= 1:
                        filtered_count += 1
                        continue

                    processed_data.append(
                        {"text": text, "label": label, "conversation": conversation}
                    )

        # Log filtering statistics
        total_original = (
            len(raw_dataset["train"])
            if len(raw_dataset) == 1 and "train" in raw_dataset
            else sum(len(split) for split in raw_dataset.values())
        )
        logger.info(
            f"Processed {len(processed_data)} examples from {total_original} original samples"
        )
        if filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} samples with ≤1 Assistant turn ({filtered_count/total_original*100:.1f}%)"
            )

        # Convert to pandas for easier manipulation
        df = pd.DataFrame(processed_data)

        # Sanity check: estimate token lengths and truncation risk before splitting
        self._log_length_stats(df["text"].tolist())

        # Create validation and test sets with similar label distribution to training set
        # First, calculate overall label distribution
        total_samples = len(df)
        label_dist = df["label"].value_counts(normalize=True)
        logger.info(f"Overall label distribution: {label_dist.to_dict()}")

        # Set validation and test set sizes (as percentages of total data)
        val_ratio = self.val_ratio  # Use configurable validation ratio
        test_ratio = self.test_ratio  # Use configurable test ratio

        val_size = int(total_samples * val_ratio)
        test_size = int(total_samples * test_ratio)

        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Validation size: {val_size}, Test size: {test_size}")

        # Calculate number of samples per class for validation and test
        # to maintain similar distribution to the overall dataset
        val_samples_per_class = {}
        test_samples_per_class = {}

        for label in df["label"].unique():
            label_proportion = label_dist[label]
            val_samples_per_class[label] = int(val_size * label_proportion)
            test_samples_per_class[label] = int(test_size * label_proportion)

        logger.info(f"Validation samples per class: {val_samples_per_class}")
        logger.info(f"Test samples per class: {test_samples_per_class}")

        # Separate by class
        df_by_label = {
            label: df[df["label"] == label] for label in df["label"].unique()
        }

        # Check if we have enough samples per class
        for label, count in val_samples_per_class.items():
            available = len(df_by_label[label])
            if count > available:
                logger.warning(
                    f"Requested {count} samples for label {label} in validation, but only {available} available. Using all available."
                )
                val_samples_per_class[label] = available

        for label, count in test_samples_per_class.items():
            available = len(df_by_label[label])
            required_for_val = val_samples_per_class[label]
            if count > (available - required_for_val):
                adjusted = max(0, available - required_for_val)
                logger.warning(
                    f"Requested {count} samples for label {label} in test, but only {adjusted} available after validation. Using {adjusted}."
                )
                test_samples_per_class[label] = adjusted

        # Sample for validation (maintaining distribution)
        val_dfs = []
        for label, count in val_samples_per_class.items():
            if count > 0:
                val_sample = df_by_label[label].sample(n=count, random_state=42)
                val_dfs.append(val_sample)

        val_df = (
            pd.concat(val_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        )

        # Sample for test (maintaining distribution) from remaining samples
        test_dfs = []
        # Collect indices of samples used in validation
        val_indices_used = pd.Index([])
        if len(val_dfs) > 0:
            for val_sample in val_dfs:
                val_indices_used = val_indices_used.union(val_sample.index)

        for label, count in test_samples_per_class.items():
            if count > 0:
                remaining_for_label = df_by_label[label].drop(
                    val_indices_used, errors="ignore"
                )
                if len(remaining_for_label) >= count:
                    test_sample = remaining_for_label.sample(n=count, random_state=43)
                    test_dfs.append(test_sample)
                else:
                    logger.warning(
                        f"Only {len(remaining_for_label)} samples remaining for label {label} in test set"
                    )
                    if len(remaining_for_label) > 0:
                        test_dfs.append(remaining_for_label)

        test_df = (
            pd.concat(test_dfs).sample(frac=1, random_state=43).reset_index(drop=True)
            if test_dfs
            else pd.DataFrame()
        )

        # Use remaining samples for training (exclude validation AND test indices)
        # Collect all indices used in validation and test sets
        all_used_indices = val_indices_used.copy()
        if len(test_dfs) > 0:
            for test_sample in test_dfs:
                all_used_indices = all_used_indices.union(test_sample.index)

        train_df = df.drop(all_used_indices).reset_index(drop=True)

        logger.info(
            f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}"
        )

        # Calculate and log detailed distribution information
        train_dist = train_df["label"].value_counts(normalize=True).sort_index()
        val_dist = (
            val_df["label"].value_counts(normalize=True).sort_index()
            if len(val_df) > 0
            else pd.Series()
        )
        test_dist = (
            test_df["label"].value_counts(normalize=True).sort_index()
            if len(test_df) > 0
            else pd.Series()
        )

        logger.info("=" * 60)
        logger.info("DATASET SPLIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Validation size: {len(val_df)}")
        logger.info(f"Test size: {len(test_df)}")
        logger.info("")

        logger.info("LABEL DISTRIBUTIONS (proportions):")
        logger.info(f"Train:      {train_dist.to_dict()}")
        if len(val_df) > 0:
            logger.info(f"Validation: {val_dist.to_dict()}")
        if len(test_df) > 0:
            logger.info(f"Test:       {test_dist.to_dict()}")
        logger.info("")

        logger.info("LABEL DISTRIBUTIONS (counts):")
        logger.info(
            f"Train:      {train_df['label'].value_counts().sort_index().to_dict()}"
        )
        if len(val_df) > 0:
            logger.info(
                f"Validation: {val_df['label'].value_counts().sort_index().to_dict()}"
            )
        if len(test_df) > 0:
            logger.info(
                f"Test:       {test_df['label'].value_counts().sort_index().to_dict()}"
            )

        # Check distribution similarity
        if len(val_df) > 0 and len(test_df) > 0:
            logger.info("")
            logger.info("DISTRIBUTION SIMILARITY CHECK:")
            # Calculate differences between train and val/test distributions
            for label in train_dist.index:
                train_prop = train_dist[label]
                val_prop = val_dist.get(label, 0)
                test_prop = test_dist.get(label, 0)
                val_diff = abs(train_prop - val_prop)
                test_diff = abs(train_prop - test_prop)
                logger.info(
                    f"Label {label}: Train={train_prop:.3f}, Val={val_prop:.3f} (diff={val_diff:.3f}), Test={test_prop:.3f} (diff={test_diff:.3f})"
                )

        logger.info("=" * 60)

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        return DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

    def format_conversation(self, conversation: List[str]) -> str:
        """Convert a prompt_answer-style string into Assistant/Student lines without altering content."""

        return "\n\n".join(conversation[:-1])

    def _log_length_stats(self, texts: List[str]) -> None:
        """Tokenize without truncation to estimate sequence lengths and truncation risk."""
        try:
            # Create a temporary tokenizer copy to avoid affecting the main one
            temp_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if temp_tokenizer.pad_token is None:
                if temp_tokenizer.eos_token is not None:
                    temp_tokenizer.pad_token = temp_tokenizer.eos_token
                    temp_tokenizer.pad_token_id = temp_tokenizer.eos_token_id
                else:
                    temp_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            lengths: List[int] = []
            # Batch to avoid OOM
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = temp_tokenizer(
                    batch,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )
                if isinstance(enc["input_ids"][0], list):
                    lengths.extend(len(x) for x in enc["input_ids"])
                else:
                    lengths.extend([len(enc["input_ids"])])
            arr = np.array(lengths)
            over = (arr > self.max_length).mean() if len(arr) else 0.0
            logger.info(
                f"Token length stats: min={arr.min() if len(arr) else 0}, p50={np.median(arr) if len(arr) else 0}, p90={np.percentile(arr,90) if len(arr) else 0}, max={arr.max() if len(arr) else 0}; >max_length({self.max_length})={over:.1%}"
            )
        except Exception as e:
            logger.warning(f"Could not compute token length stats: {e}")

    def tokenize_function(self, examples):
        """Tokenize the conversations"""
        # Tokenize the text - remove padding here since we'll use DataCollatorWithPadding
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Don't pad here - let the data collator handle it
            max_length=self.max_length,
            return_tensors=None,  # Return as lists, not tensors
        )

        # Add labels as integers (0 or 1) for classification
        tokenized["labels"] = [int(label) for label in examples["label"]]

        return tokenized

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # logits shape [N, 2]
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            exp = np.exp(predictions - predictions.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            pos_probs = probs[:, 1]
            preds = probs.argmax(axis=1)
        else:
            pos_probs = 1.0 / (1.0 + np.exp(-predictions.reshape(-1)))
            preds = (pos_probs > 0.5).astype(int)

        labels = labels.reshape(-1)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(labels, pos_probs)
        except ValueError:
            auc = 0.5
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

    def train(self, dataset: DatasetDict):
        """Train the reward model"""
        logger.info("Starting training...")

        # Tokenize datasets - remove ALL non-essential columns
        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[
                col for col in dataset["train"].column_names if col not in ["labels"]
            ],
        )

        # Optional sanity checks
        logger.info("Dataset features after tokenization:")
        for split in ["train", "validation", "test"]:
            ds = tokenized_datasets[split]
            logger.info(f"{split}: {ds.features}")
            # Check a few label values
            sample_labels = ds["labels"][:5]
            logger.info(f"{split} sample labels: {sample_labels}")

        # Create data collator for dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True, return_tensors="pt"
        )

        # Training arguments - configured for model parallelism with stability fixes
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,  # Evaluate even less frequently
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,  # Enable to save memory and stabilize
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            and torch.cuda.is_available(),
            report_to="wandb" if wandb.run else None,
            remove_unused_columns=False,
            run_name="llama3_1_8b_reward_model_stable",
            # Disable data parallelism since we're using model parallelism
            dataloader_num_workers=0,
            local_rank=-1,
            ddp_find_unused_parameters=False,
            # Add stability parameters
            max_grad_norm=1.0,  # Gradient clipping to prevent explosion
            lr_scheduler_type="cosine",  # Cosine scheduler for stable learning
            # warmup_steps=100,  # More gradual warmup
            save_total_limit=3,  # Keep only best models
            eval_accumulation_steps=1,  # Reduce memory pressure
        )

        # Custom trainer class to handle model parallelism
        class RewardTrainer(Trainer):
            def _move_model_to_device(self, model, device):
                """Override to prevent automatic device movement since we use device_map"""
                # Do nothing - model is already placed via device_map="auto"
                return model

            def _wrap_model(self, model, training=True, dataloader=None):
                """Override to prevent DataParallel wrapping since we use model parallelism"""
                # Return model as-is, don't wrap in DataParallel
                return model

        # Initialize trainer with data collator
        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        # Evaluate on test set and make the logging clearer
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(
            tokenized_datasets["test"], metric_key_prefix="test"
        )

        # Log with clearer distinction
        logger.info("=" * 50)
        logger.info("FINAL RESULTS SUMMARY:")
        logger.info("=" * 50)
        logger.info(f"Test Accuracy: {test_results.get('test_accuracy', 'N/A'):.3f}")
        logger.info(f"Test F1 Score: {test_results.get('test_f1', 'N/A'):.3f}")
        logger.info(f"Test AUC: {test_results.get('test_auc', 'N/A'):.3f}")
        logger.info(f"Test Precision: {test_results.get('test_precision', 'N/A'):.3f}")
        logger.info(f"Test Recall: {test_results.get('test_recall', 'N/A'):.3f}")
        logger.info(f"Full test results: {test_results}")
        logger.info("=" * 50)

        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        return trainer, test_results

    def inference_rerank(
        self, conversations: List[str], top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Rerank multiple candidate conversations based on reward scores
        Uses the probability of the positive class (label 1) as the score
        """
        self.model.eval()
        scores = []

        with torch.no_grad():
            for conv in conversations:
                # Tokenize with the same padding configuration as training
                inputs = self.tokenizer(
                    conv,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )

                # Move to appropriate device (first GPU for input)
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Convert logits to probabilities and get probability of positive class (label 1)
                probs = F.softmax(logits, dim=-1)
                positive_prob = probs[0, 1].item()  # Probability of label 1

                scores.append(positive_prob)

        # Sort by score (higher probability of positive class is better)
        ranked_conversations = sorted(
            zip(conversations, scores), key=lambda x: x[1], reverse=True
        )

        return ranked_conversations[:top_k]


def main():
    """Main training function"""
    # Initialize wandb for logging (optional)
    wandb.init(
        project="reward-model-training",
        name="llama3.1-8b-acc-reward-model-1e5-bz32",
        config={
            "model_name": "meta-llama/Llama-3.1-8B",
            "max_length": 2048,
            "learning_rate": 5e-6,
            "batch_size": 32,
            "num_epochs": 3,
            "model_parallelism": True,
            "device_map": "auto",
            "model_type": "AutoModelForSequenceClassification",
            "num_labels": 2,
        },
    )

    # Initialize trainer
    trainer = RewardModelTrainer(
        model_name="meta-llama/Llama-3.1-8B",
        max_length=2048,
        learning_rate=5e-6,  # Much lower learning rate
        batch_size=32,
        num_epochs=3,  # Reduce epochs
        output_dir="./llama_acc_reward_model_1e5-bz32",
    )

    # Load and process dataset
    dataset = trainer.load_and_process_dataset("Jennny/final_acc_rate_latest3")

    # Train the model
    trained_model, test_results = trainer.train(dataset)

    logger.info("Training completed!")
    logger.info(f"Final test results: {test_results}")


if __name__ == "__main__":
    main()
