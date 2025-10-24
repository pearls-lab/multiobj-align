########################
# This script is modified from DPO codebase https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
########################
import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    slice_and_move_batch_for_device_balanced,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = 0
    logits = pi_logratios - ref_logratios
    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )
    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
) -> Dict[str, torch.LongTensor]:
    max_length = max(
        batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
    )
    concatenated_batch = {}
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(
                batch[k], max_length, pad_value=pad_value
            )
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )
    return concatenated_batch


class BasicTrainer(object):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        tokenizer_name_or_path = (
            config.model.tokenizer_name_or_path or config.model.name_or_path
        )
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == "sft",
            model_name=config.model.name_or_path,
            max_samples=getattr(config, "max_samples", 10000),
        )
        self.policy = policy
        self.reference_model = reference_model
        self.train_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split="train",
            n_epochs=config.n_epochs,
            n_examples=config.n_examples,
            batch_size=config.batch_size,
            silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs),
        )
        rank0_print(f"Loaded train data iterator")
        self.eval_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split="test",
            n_examples=config.n_eval_examples,
            batch_size=config.eval_batch_size,
            silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs),
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        ctx = lambda: (
            FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if self.config.loss.name in {"dpo", "ipo"}:
            ctx = lambda: (
                FSDP.summon_full_params(
                    self.reference_model, writeback=False, recurse=False
                )
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )
        if self.config.loss.name in {"dpo", "ipo"}:
            reference_output = pad_to_length(
                reference_output, self.config.max_length, self.tokenizer.pad_token_id
            )
            reference_output = all_gather_if_needed(
                reference_output, self.rank, self.world_size
            )
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )
        else:
            reference_output_decoded = []
        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = concatenated_inputs(batch)
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        )
        all_logits = outputs["logits"].to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: DictConfig,
        train=True,
    ):
        """Calculate metrics for a batch of data using multiple heads efficiently."""
        if self.batch_counter == 0:
            rank0_print("DIMENSION TO DATASET MAPPING:")
            for dim_idx, dataset_name in enumerate(self.config.datasets):
                rank0_print(f"Dimension {dim_idx} -> Dataset: {dataset_name}")
        metrics = {}
        train_test = "train" if train else "eval"
        num_heads = len(self.config.datasets)
        batch_size = batch["chosen_input_ids"].shape[0]

        # Get dimension information for each example
        dimensions = batch.get(
            "dimensions", list(range(num_heads)) * (batch_size // num_heads + 1)
        )
        dimensions = dimensions[:batch_size]  # Truncate to batch size

        # Create separate mini-batches for each dimension and ensure non-empty
        dim_batches = {dim_idx: {"indices": []} for dim_idx in range(num_heads)}
        for i, dim_idx in enumerate(dimensions):
            if dim_idx < num_heads:
                dim_batches[dim_idx]["indices"].append(i)

        # Skip empty dimensions - add debugging
        for dim_idx in range(num_heads):
            dim_indices = dim_batches[dim_idx]["indices"]
            dim_name = self.config.datasets[dim_idx]
            metrics[f"{dim_name}/examples_seen"] = len(dim_indices)

            if not dim_indices:
                rank0_print(
                    f"WARNING: No examples for dimension {dim_idx} ({dim_name}) in batch"
                )
                # Initialize empty metrics to ensure consistent tracking
                metrics[f"{dim_name}/loss/{train_test}"] = []
                metrics[f"{dim_name}/rewards_{train_test}/chosen"] = []
                metrics[f"{dim_name}/rewards_{train_test}/rejected"] = []
                metrics[f"{dim_name}/rewards_{train_test}/accuracies"] = []
                metrics[f"{dim_name}/rewards_{train_test}/margins"] = []
                metrics[f"{dim_name}/logps_{train_test}/chosen"] = []
                metrics[f"{dim_name}/logps_{train_test}/rejected"] = []
                continue

        # Forward pass through policy model
        policy_outputs = self.policy(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        policy_chosen_logits = policy_outputs["logits"]

        rejected_policy_outputs = self.policy(
            batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        policy_rejected_logits = rejected_policy_outputs["logits"]

        # Forward pass through reference model (no gradient)
        with torch.no_grad():
            ref_outputs = self.reference_model(
                batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            ref_chosen_logits = ref_outputs["logits"]

            ref_rejected_outputs = self.reference_model(
                batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )
            ref_rejected_logits = ref_rejected_outputs["logits"]

        # Compute losses for each dimension
        combined_losses = []

        for dim_idx in range(num_heads):
            indices = dim_batches[dim_idx]["indices"]
            if not indices:
                continue  # Skip empty dimensions

            dim_name = self.config.datasets[dim_idx]

            try:
                # Extract logits for this dimension's head
                dim_policy_chosen_logits = policy_chosen_logits[dim_idx, indices]
                dim_policy_rejected_logits = policy_rejected_logits[dim_idx, indices]
                dim_ref_chosen_logits = ref_chosen_logits[dim_idx, indices]
                dim_ref_rejected_logits = ref_rejected_logits[dim_idx, indices]

                # Extract corresponding labels
                dim_chosen_labels = batch["chosen_labels"][indices]
                dim_rejected_labels = batch["rejected_labels"][indices]

                # Compute log probabilities
                dim_policy_chosen_logps = _get_batch_logps(
                    dim_policy_chosen_logits, dim_chosen_labels, average_log_prob=False
                )

                dim_policy_rejected_logps = _get_batch_logps(
                    dim_policy_rejected_logits,
                    dim_rejected_labels,
                    average_log_prob=False,
                )

                dim_ref_chosen_logps = _get_batch_logps(
                    dim_ref_chosen_logits, dim_chosen_labels, average_log_prob=False
                )

                dim_ref_rejected_logps = _get_batch_logps(
                    dim_ref_rejected_logits, dim_rejected_labels, average_log_prob=False
                )

                # Prepare loss arguments
                if loss_config.name == "dpo":
                    loss_kwargs = {
                        "beta": loss_config.beta,
                        "reference_free": getattr(loss_config, "reference_free", False),
                        "label_smoothing": getattr(loss_config, "label_smoothing", 0.0),
                        "ipo": False,
                    }
                elif loss_config.name == "ipo":
                    loss_kwargs = {"beta": loss_config.beta, "ipo": True}
                else:
                    raise ValueError(f"unknown loss {loss_config.name}")

                # Calculate preference loss for this dimension
                dim_losses, dim_chosen_rewards, dim_rejected_rewards = preference_loss(
                    dim_policy_chosen_logps,
                    dim_policy_rejected_logps,
                    dim_ref_chosen_logps,
                    dim_ref_rejected_logps,
                    **loss_kwargs,
                )

                # Add to combined losses
                if dim_losses.numel() > 0:
                    # Calculate mean loss for this dimension
                    dim_loss = dim_losses.mean()
                    combined_losses.append(dim_loss)

                    # Calculate metrics
                    dim_reward_accuracies = (
                        dim_chosen_rewards > dim_rejected_rewards
                    ).float()
                    dim_margins = dim_chosen_rewards - dim_rejected_rewards

                    # Add dimension-specific metrics
                    metrics[f"{dim_name}/loss/{train_test}"] = (
                        all_gather_if_needed(
                            dim_losses.detach(), self.rank, self.world_size
                        )
                        .cpu()
                        .float()
                        .numpy()
                        .tolist()
                    )

                    metrics[f"{dim_name}/rewards_{train_test}/chosen"] = (
                        all_gather_if_needed(
                            dim_chosen_rewards, self.rank, self.world_size
                        )
                        .cpu()
                        .float()
                        .numpy()
                        .tolist()
                    )

                    metrics[f"{dim_name}/rewards_{train_test}/rejected"] = (
                        all_gather_if_needed(
                            dim_rejected_rewards, self.rank, self.world_size
                        )
                        .cpu()
                        .float()
                        .numpy()
                        .tolist()
                    )

                    metrics[f"{dim_name}/rewards_{train_test}/accuracies"] = (
                        all_gather_if_needed(
                            dim_reward_accuracies, self.rank, self.world_size
                        )
                        .cpu()
                        .float()
                        .numpy()
                        .tolist()
                    )

                    metrics[f"{dim_name}/rewards_{train_test}/margins"] = (
                        all_gather_if_needed(dim_margins, self.rank, self.world_size)
                        .cpu()
                        .float()
                        .numpy()
                        .tolist()
                    )

                    metrics[f"{dim_name}/logps_{train_test}/chosen"] = (
                        all_gather_if_needed(
                            dim_policy_chosen_logps.detach(), self.rank, self.world_size
                        )
                        .cpu()
                        .float()
                        .numpy()
                        .tolist()
                    )

                    metrics[f"{dim_name}/logps_{train_test}/rejected"] = (
                        all_gather_if_needed(
                            dim_policy_rejected_logps.detach(),
                            self.rank,
                            self.world_size,
                        )
                        .cpu()
                        .float()
                        .numpy()
                        .tolist()
                    )

                    # Calculate mean metrics for easy logging
                    if train:
                        metrics[f"{dim_name}/mean_loss"] = dim_losses.mean().item()
                        metrics[f"{dim_name}/mean_accuracy"] = (
                            dim_reward_accuracies.mean().item()
                        )

            except Exception as e:
                rank0_print(
                    f"Error computing metrics for dimension {dim_idx} ({dim_name}): {e}"
                )
                # Add error information to metrics
                metrics[f"{dim_name}/error"] = str(e)

        # Calculate combined loss (average of all dimension losses)
        if combined_losses:
            combined_loss = torch.stack(combined_losses).mean()
        else:
            # Default loss if no examples
            combined_loss = torch.tensor(
                0.0, device=self.policy.device, requires_grad=True
            )

        # Add combined loss to metrics
        metrics[f"loss/{train_test}"] = (
            all_gather_if_needed(combined_loss.detach(), self.rank, self.world_size)
            .cpu()
            .float()
            .numpy()
            .tolist()
        )

        return combined_loss, metrics

    def train(self):
        """Train the model with specialized heads for each dimension."""
        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {"dpo", "ipo"}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        # Add tracking for per-dimension metrics
        dimension_counters = {dim: 0 for dim in self.config.datasets}
        dimension_losses = {dim: [] for dim in self.config.datasets}
        dimension_accuracies = {dim: [] for dim in self.config.datasets}

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                rank0_print(
                    f"Running evaluation after {self.example_counter} train examples"
                )
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(
                        columns=["step", "prompt", "sample", "dimension"]
                    )
                    if self.config.loss.name in {"dpo", "ipo"}:
                        reference_text_table = wandb.Table(
                            columns=["step", "prompt", "sample", "dimension"]
                        )

                for eval_batch in (
                    tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                    if self.rank == 0
                    else self.eval_batches
                ):
                    local_eval_batch = slice_and_move_batch_for_device_balanced(
                        eval_batch,
                        self.rank,
                        self.world_size,
                        self.rank,
                        preserve_dimensions=True,
                    )
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch,
                            self.config.loss,
                            train=False,
                            # preserve_dimensions=True,
                        )

                    for k, v in eval_metrics.items():
                        if isinstance(v, list):
                            all_eval_metrics[k].extend(v)
                        else:
                            all_eval_metrics[k].append(v)

                if self.config.sample_during_eval:
                    # Sampling code remains similar but adds dimension info
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(
                            f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts."
                        )
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = (
                            self.config.n_eval_model_samples
                            // self.config.eval_batch_size
                        )
                        sample_batches = self.eval_batches[:n_sample_batches]

                    for eval_batch in (
                        tqdm.tqdm(sample_batches, desc="Generating samples...")
                        if self.rank == 0
                        else sample_batches
                    ):
                        local_eval_batch = slice_and_move_batch_for_device_balanced(
                            eval_batch,
                            self.rank,
                            self.world_size,
                            self.rank,
                            preserve_dimensions=True,
                        )
                        policy_samples, reference_samples = self.get_batch_samples(
                            local_eval_batch
                        )

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        # Get dimensions for each example
                        batch_dims = local_eval_batch.get(
                            "dimensions", [0] * len(policy_samples)
                        )

                        for i, (prompt, sample) in enumerate(
                            zip(eval_batch["prompt"], policy_samples)
                        ):
                            dim_idx = batch_dims[i] if i < len(batch_dims) else 0
                            dim_name = self.config.datasets[dim_idx]
                            policy_text_table.add_data(
                                self.example_counter, prompt, sample, dim_name
                            )

                        if self.config.loss.name in {"dpo", "ipo"}:
                            for i, (prompt, sample) in enumerate(
                                zip(eval_batch["prompt"], reference_samples)
                            ):
                                dim_idx = batch_dims[i] if i < len(batch_dims) else 0
                                dim_name = self.config.datasets[dim_idx]
                                reference_text_table.add_data(
                                    self.example_counter, prompt, sample, dim_name
                                )

                # Calculate and log metrics
                mean_eval_metrics = {
                    k: sum(v) / len(v)
                    for k, v in all_eval_metrics.items()
                    if isinstance(v, list) and len(v) > 0
                }
                # For non-list metrics, simply copy them
                for k, v in all_eval_metrics.items():
                    if not isinstance(v, list):
                        mean_eval_metrics[k] = v

                rank0_print(
                    f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                )

                if self.config.sample_during_eval:
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {"dpo", "ipo"}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log(
                            {"policy_samples": policy_text_table},
                            step=self.example_counter,
                        )
                        if self.config.loss.name in {"dpo", "ipo"}:
                            wandb.log(
                                {"reference_samples": reference_text_table},
                                step=self.example_counter,
                            )

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print("skipping save in debug mode")
                    else:
                        output_dir = os.path.join(
                            self.run_dir, f"step-{self.example_counter}"
                        )
                        rank0_print(f"creating checkpoint to write to {output_dir}...")
                        self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)

            # Update dimension-specific counters
            if "dimensions" in batch:
                for dim_idx in batch["dimensions"]:
                    if dim_idx < len(self.config.datasets):
                        dimension_counters[self.config.datasets[dim_idx]] += 1

            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device_balanced(
                    batch,
                    microbatch_idx,
                    self.config.gradient_accumulation_steps,
                    self.rank,
                    preserve_dimensions=True,
                )
                local_microbatch = slice_and_move_batch_for_device_balanced(
                    global_microbatch,
                    self.rank,
                    self.world_size,
                    self.rank,
                    preserve_dimensions=True,
                )
                loss, metrics = self.get_batch_metrics(
                    local_microbatch, self.config.loss, train=True
                )
                (loss / self.config.gradient_accumulation_steps).backward()

                # Track dimension-specific metrics
                for dim_name in self.config.datasets:
                    dim_loss_key = f"{dim_name}/loss/train"
                    dim_acc_key = f"{dim_name}/rewards_train/accuracies"

                    if (
                        dim_loss_key in metrics
                        and isinstance(metrics[dim_loss_key], list)
                        and len(metrics[dim_loss_key]) > 0
                    ):
                        dimension_losses[dim_name].extend(metrics[dim_loss_key])

                    if (
                        dim_acc_key in metrics
                        and isinstance(metrics[dim_acc_key], list)
                        and len(metrics[dim_acc_key]) > 0
                    ):
                        dimension_accuracies[dim_name].extend(metrics[dim_acc_key])

                for k, v in metrics.items():
                    if isinstance(v, list):
                        batch_metrics[k].extend(v)
                    else:
                        batch_metrics[k].append(v)

            # Apply gradients and update model
            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)
            batch_metrics["grad_norm"].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {
                    k: sum(v) / len(v)
                    for k, v in batch_metrics.items()
                    if isinstance(v, list) and len(v) > 0
                }
                # For non-list metrics, simply copy them
                for k, v in batch_metrics.items():
                    if not isinstance(v, list):
                        mean_train_metrics[k] = v

                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter

                # Add dimension-specific metrics
                for dim_name in self.config.datasets:
                    mean_train_metrics[f"{dim_name}/examples_seen"] = (
                        dimension_counters[dim_name]
                    )
                    if dimension_losses[dim_name]:
                        mean_train_metrics[f"{dim_name}/mean_loss"] = sum(
                            dimension_losses[dim_name]
                        ) / len(dimension_losses[dim_name])
                    if dimension_accuracies[dim_name]:
                        mean_train_metrics[f"{dim_name}/mean_accuracy"] = sum(
                            dimension_accuracies[dim_name]
                        ) / len(dimension_accuracies[dim_name])

                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

                # Reset dimension-specific metrics
                dimension_losses = {dim: [] for dim in self.config.datasets}
                dimension_accuracies = {dim: [] for dim in self.config.datasets}
            else:
                rank0_print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )
            #### END TRAINING ####

    def clip_gradient(self):
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f"LATEST")
        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter, policy_state_dict, metrics, "policy.pt", output_dir
        )
        del policy_state_dict
        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(
            self.example_counter,
            optimizer_state_dict,
            metrics,
            "optimizer.pt",
            output_dir,
        )
        del optimizer_state_dict
        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(
            self.example_counter,
            scheduler_state_dict,
            metrics,
            "scheduler.pt",
            output_dir,
        )


### Add to trainers_math2.py ###
class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer subclass that uses PyTorch FSDP to shard the multi-head model across multiple GPUs."""

        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )
        assert (
            config.model.block_name is not None
        ), "must specify model.block_name (e.g., GPT2Block or Qwen2DecoderLayer) for FSDP"

        # Find transformer block class to use for auto-wrapping policy
        wrap_class = get_block_class_from_model(
            policy.base_model, config.model.block_name
        )
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        # Add support for controlling head wrapping
        ignored_modules = None
        if hasattr(policy, "heads") and not getattr(
            config.model, "fsdp_wrap_heads", True
        ):
            ignored_modules = policy.heads
            rank0_print("Not wrapping heads with FSDP")

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=ignored_modules,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy using FSDP...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(
            policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy
        )

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")
                apply_activation_checkpointing(
                    self.policy,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn,
                )
                rank0_print("FSDP activation checkpointing enabled!")

        # Also shard reference model if in DPO mode
        if config.loss.name in {"dpo", "ipo"}:
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print(f"FSDP initialization complete on rank {rank}")
        dist.barrier()

    def get_batch_metrics(self, batch, loss_config, train=True):
        """Override to ensure special handling for FSDP with multiple dimensions."""
        # First check if we have any examples for each dimension
        num_heads = len(self.config.datasets)

        # Get dimensions for debugging
        dimensions = batch.get("dimensions", [])

        # Count examples per dimension
        dim_counts = defaultdict(int)
        for dim_idx in dimensions:
            if dim_idx < num_heads:
                dim_counts[dim_idx] += 1

        # Log dimension counts in this batch
        for dim_idx in range(num_heads):
            dim_name = self.config.datasets[dim_idx]
            rank0_print(
                f"Rank {self.rank} - dimension {dim_idx} ({dim_name}) has {dim_counts[dim_idx]} examples in batch"
            )

        # Call parent implementation
        return super().get_batch_metrics(batch, loss_config, train)

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        # Set up FSDP state dict config for saving the full model on rank 0
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                "policy.pt",
                output_dir,
            )
        del policy_state_dict
        dist.barrier()

        # Save optimizer state (important for continuing training)
        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy,
            StateDictType.FULL_STATE_DICT,
            optim_state_dict_config=save_policy,
        ):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                optimizer_state_dict,
                metrics,
                "optimizer.pt",
                output_dir,
            )
        del optimizer_state_dict
        dist.barrier()

        # Save scheduler state
        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                self.example_counter,
                scheduler_state_dict,
                metrics,
                "scheduler.pt",
                output_dir,
            )
        dist.barrier()

    def train(self):
        """Train the model with specialized heads for each dimension."""
        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {"dpo", "ipo"}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        # Add tracking for per-dimension metrics
        dimension_counters = {dim: 0 for dim in self.config.datasets}
        dimension_losses = {dim: [] for dim in self.config.datasets}
        dimension_accuracies = {dim: [] for dim in self.config.datasets}

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                rank0_print(
                    f"Running evaluation after {self.example_counter} train examples"
                )
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(
                        columns=["step", "prompt", "sample", "dimension"]
                    )
                    if self.config.loss.name in {"dpo", "ipo"}:
                        reference_text_table = wandb.Table(
                            columns=["step", "prompt", "sample", "dimension"]
                        )

                for eval_batch in (
                    tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
                    if self.rank == 0
                    else self.eval_batches
                ):
                    local_eval_batch = slice_and_move_batch_for_device_balanced(
                        eval_batch,
                        self.rank,
                        self.world_size,
                        self.rank,
                        preserve_dimensions=True,
                    )
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(
                            local_eval_batch, self.config.loss, train=False
                        )

                    for k, v in eval_metrics.items():
                        if isinstance(v, list):
                            all_eval_metrics[k].extend(v)
                        else:
                            all_eval_metrics[k].append(v)

                if self.config.sample_during_eval:
                    # Sampling code (abbreviated for brevity)
                    pass

                # Calculate and log metrics
                mean_eval_metrics = {
                    k: sum(v) / len(v)
                    for k, v in all_eval_metrics.items()
                    if isinstance(v, list) and len(v) > 0
                }

                # Add non-list metrics
                for k, v in all_eval_metrics.items():
                    if not isinstance(v, list):
                        mean_eval_metrics[k] = v

                rank0_print(
                    f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print("skipping save in debug mode")
                    else:
                        output_dir = os.path.join(
                            self.run_dir, f"step-{self.example_counter}"
                        )
                        rank0_print(f"creating checkpoint to write to {output_dir}...")
                        self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)

            # Update dimension-specific counters
            if "dimensions" in batch:
                for dim_idx in batch["dimensions"]:
                    if dim_idx < len(self.config.datasets):
                        dimension_counters[self.config.datasets[dim_idx]] += 1

            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device_balanced(
                    batch,
                    microbatch_idx,
                    self.config.gradient_accumulation_steps,
                    self.rank,
                    preserve_dimensions=True,
                )
                local_microbatch = slice_and_move_batch_for_device_balanced(
                    global_microbatch,
                    self.rank,
                    self.world_size,
                    self.rank,
                    preserve_dimensions=True,
                )
                loss, metrics = self.get_batch_metrics(
                    local_microbatch, self.config.loss, train=True
                )
                (loss / self.config.gradient_accumulation_steps).backward()

                # Track dimension-specific metrics
                for dim_name in self.config.datasets:
                    dim_loss_key = f"{dim_name}/loss/train"
                    dim_acc_key = f"{dim_name}/rewards_train/accuracies"

                    if (
                        dim_loss_key in metrics
                        and isinstance(metrics[dim_loss_key], list)
                        and len(metrics[dim_loss_key]) > 0
                    ):
                        dimension_losses[dim_name].extend(metrics[dim_loss_key])

                    if (
                        dim_acc_key in metrics
                        and isinstance(metrics[dim_acc_key], list)
                        and len(metrics[dim_acc_key]) > 0
                    ):
                        dimension_accuracies[dim_name].extend(metrics[dim_acc_key])

                for k, v in metrics.items():
                    if isinstance(v, list):
                        batch_metrics[k].extend(v)
                    else:
                        batch_metrics[k].append(v)

            # Apply gradients and update model
            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)
            batch_metrics["grad_norm"].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {
                    k: sum(v) / len(v)
                    for k, v in batch_metrics.items()
                    if isinstance(v, list) and len(v) > 0
                }

                # Add non-list metrics
                for k, v in batch_metrics.items():
                    if not isinstance(v, list):
                        mean_train_metrics[k] = v

                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter

                # Add dimension-specific metrics
                for dim_name in self.config.datasets:
                    mean_train_metrics[f"{dim_name}/examples_seen"] = (
                        dimension_counters[dim_name]
                    )
                    if dimension_losses[dim_name]:
                        mean_train_metrics[f"{dim_name}/mean_loss"] = sum(
                            dimension_losses[dim_name]
                        ) / len(dimension_losses[dim_name])
                    if dimension_accuracies[dim_name]:
                        mean_train_metrics[f"{dim_name}/mean_accuracy"] = sum(
                            dimension_accuracies[dim_name]
                        ) / len(dimension_accuracies[dim_name])

                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()

                # Reset dimension-specific metrics
                dimension_losses = {dim: [] for dim in self.config.datasets}
                dimension_accuracies = {dim: [] for dim in self.config.datasets}
            else:
                rank0_print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )
            #### END TRAINING ####
