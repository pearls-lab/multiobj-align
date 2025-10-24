"""
Value model training script for Qwen2.5-Math-PRM-7B with <extra_0> as step separator.
Using MSE loss with soft labels.
"""

import os

# Remove any leftover DDP env-vars so that `device_map="auto"` won't try to initialize torch.distributed
for v in ("LOCAL_RANK", "WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(v, None)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
import wandb
import logging
import argparse
from typing import List, Dict, Optional, Tuple, Union
from huggingface_hub import login, HfApi

# Set up logging to print debug messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Qwen2PrmWithValueHead(PreTrainedModel):
    """
    Qwen2PrmWithValueHead adds a value prediction head on top of the Qwen2.5-Math-PRM-7B model.

    This model takes the existing PRM model and adds a simple value head that predicts
    the expected future reward from a mathematical reasoning state.
    """

    def __init__(self, prm_model):
        """
        Initialize the model with a pre-trained PRM model and add a value head.

        Args:
            prm_model: The pre-trained Qwen2.5-Math-PRM-7B model
        """
        # Initialize with the config from the PRM model
        super().__init__(prm_model.config)

        # Save the original PRM model
        self.prm_model = prm_model

        # Store the step separator token ID
        self.step_sep_token = "<extra_0>"
        if hasattr(prm_model, "tokenizer"):
            self.step_sep_token_id = prm_model.tokenizer.convert_tokens_to_ids(
                self.step_sep_token
            )
        else:
            # Default ID - this might need to be set correctly later
            self.step_sep_token_id = None

        # Freeze the base model parameters
        for param in self.prm_model.parameters():
            param.requires_grad = False
        logger.info("Base PRM model parameters frozen (not trainable)")

        # Get the hidden size from the config
        hidden_size = self.config.hidden_size
        logger.info(f"Creating value head with hidden size: {hidden_size}")

        # Add a value prediction head for soft labels (unbounded values)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        # Initialize the value head
        self._init_weights(self.value_head)

        # Move value head to the same device as the base model
        device = next(self.prm_model.parameters()).device
        self.value_head = self.value_head.to(device=device, dtype=torch.float32)
        logger.info(f"Value head moved to device: {device} with dtype: torch.float32")

    def _init_weights(self, module):
        """Initialize the weights of the value head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for efficient decoding
            inputs_embeds: Pre-computed input embeddings
            labels: Target value scores for regression (list of tensors for each example)
            use_cache: Whether to use the cache for efficient decoding
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary or tuple

        Returns:
            A dictionary containing loss and value predictions
        """
        # Pass inputs through the PRM model to get the base representations
        outputs = self.prm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states
            return_dict=True,  # Always use return dict for consistency
        )
        hidden_states = outputs.hidden_states[-1]

        # Convert hidden states to the same dtype as the value head
        value_head_dtype = next(self.value_head.parameters()).dtype
        hidden_states = hidden_states.to(dtype=value_head_dtype)

        # Get batch size
        batch_size = hidden_states.shape[0]

        # Extract all <extra_0> tokens for each example
        step_sep_token_id = self.step_sep_token_id
        token_masks = input_ids == step_sep_token_id

        # Process step representations and make predictions for each step
        all_step_values = []

        for i in range(batch_size):
            # Find positions of <extra_0> tokens
            extra_token_positions = torch.where(token_masks[i])[0]

            if len(extra_token_positions) > 0:
                # Extract hidden states at all step positions
                step_hidden_states = hidden_states[i, extra_token_positions]
                # Get value predictions for each step
                step_values = self.value_head(step_hidden_states)
                all_step_values.append(step_values)
            else:
                # Fallback if no step tokens found
                last_token_idx = min(
                    attention_mask[i].sum().item() - 1, hidden_states.shape[1] - 1
                )
                last_hidden = hidden_states[i, last_token_idx].unsqueeze(0)
                step_values = self.value_head(last_hidden)
                all_step_values.append(step_values)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()

            # Match predictions with labels for each example in batch
            batch_losses = []
            for i, (step_values, label_tensor) in enumerate(
                zip(all_step_values, labels)
            ):
                if isinstance(label_tensor, torch.Tensor) and label_tensor.numel() > 0:
                    # If label is a sequence, match length with predictions
                    pred_len = step_values.shape[0]
                    label_len = label_tensor.shape[0]

                    # Trim to shorter length
                    min_len = min(pred_len, label_len)
                    if min_len > 0:
                        # Calculate MSE loss for this example
                        example_loss = loss_fct(
                            step_values[:min_len].squeeze(-1),
                            label_tensor[:min_len].to(device=step_values.device),
                        )
                        batch_losses.append(example_loss)

            # Average losses across batch
            if batch_losses:
                loss = torch.stack(batch_losses).mean()

        # Return dictionary with loss and value predictions
        return {
            "loss": loss,
            "value": all_step_values,  # Return all step values
            "prm_outputs": outputs,
        }

    def get_prm_scores(self, input_ids, attention_mask=None, step_sep_token_id=None):
        """
        Get the original PRM scores for the process quality.

        This is a pass-through to the original PRM model's scoring function.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            step_sep_token_id: Token ID for step separators

        Returns:
            PRM scores for process quality
        """
        # Get the original PRM logits
        outputs = self.prm_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # The first element in the outputs tuple is the logits
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]

        # Create token masks based on step separator token
        token_masks = (
            (input_ids == step_sep_token_id) if step_sep_token_id is not None else None
        )

        if token_masks is None:
            return None

        # Calculate process quality scores using the same function as in the original code
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(
            -1
        )  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[
                :, 1
            ]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)

        return all_scores_res

    def get_value(self, input_ids, attention_mask=None):
        """
        Get the value prediction for the given input.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            The predicted value
        """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["value"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained PRM model and add a value head on top.

        Args:
            pretrained_model_name_or_path: Name or path of the pretrained PRM model

        Returns:
            An instance of Qwen2PrmWithValueHead
        """
        # Load the original PRM model
        prm_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Create the combined model
        model = cls(prm_model)

        # Load the value head weights if they exist
        value_head_path = kwargs.get("value_head_path", None)
        if value_head_path is None:
            value_head_path = os.path.join(
                pretrained_model_name_or_path, "value_head.pt"
            )

        if os.path.exists(value_head_path):
            try:
                value_head_state_dict = torch.load(value_head_path, map_location="cpu")
                model.value_head.load_state_dict(value_head_state_dict)
                logger.info(f"Loaded value head weights from {value_head_path}")
            except Exception as e:
                logger.warning(f"Error loading value head weights: {e}")

        return model

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model and its configuration.

        Args:
            save_directory: Directory to save the model to
        """
        # Save the PRM model
        self.prm_model.save_pretrained(save_directory, **kwargs)

        # Save the value head separately
        torch.save(self.value_head.state_dict(), f"{save_directory}/value_head.pt")

        # Add information about the value head to the config
        config = self.prm_model.config.to_dict()
        config["value_head"] = {
            "layers": [
                {
                    "type": "linear",
                    "in_features": self.config.hidden_size,
                    "out_features": self.config.hidden_size,
                },
                {"type": "relu"},
                {
                    "type": "linear",
                    "in_features": self.config.hidden_size,
                    "out_features": 1,
                },
            ]
        }

        # Save the updated config
        with open(f"{save_directory}/config.json", "w") as f:
            import json

            json.dump(config, f, indent=2)


class MathValueDataset(Dataset):
    """
    Dataset for training the value head on mathematical reasoning steps.

    This dataset processes examples from the math500_segregated_results.json format
    and creates training samples with states and target value scores.
    """

    def __init__(self, examples, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.samples = []
        self.max_length = max_length
        self.system_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        self.step_sep_token = "<extra_0>"  # Token used to separate steps

        # Process examples into training samples
        for example in tqdm(examples, desc="Creating dataset samples"):
            self._process_example(example)

    def _process_example(self, example):
        """Process an example into state-value pairs for different reasoning steps."""
        question = example.get("question", "")
        steps = example.get("response_steps", [])
        step_values = example.get("step_value", [])

        if not steps or not step_values or len(steps) != len(step_values):
            # Skip examples with missing data or mismatched steps and values
            return

        # Process the entire solution at once with all steps
        # Join all steps with the separator token
        state = self.step_sep_token.join(steps) + self.step_sep_token

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": state},
        ]

        conversation = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Create one sample with all step values
        self.samples.append(
            {
                "conversation": conversation,
                "target_values": step_values,
                "total_steps": len(steps),
            }
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Encode the conversation
        encoded = self.tokenizer(
            sample["conversation"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Return target_values as a tensor
        return {
            "input_ids": encoded.input_ids.squeeze(),
            "attention_mask": encoded.attention_mask.squeeze(),
            "target_values": torch.tensor(sample["target_values"], dtype=torch.float),
            "total_steps": sample["total_steps"],
        }


#########################
# Custom Collation Function #
#########################


def custom_collate_fn(batch):
    """
    Custom collation function to handle variable-length target values.

    Args:
        batch: List of samples from the dataset

    Returns:
        Dictionary with batched tensors and lists
    """
    # Extract items from the batch
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Handle target_values separately since they have different lengths
    target_values = [item["target_values"] for item in batch]

    # Other fields
    total_steps = torch.tensor([item["total_steps"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_values": target_values,  # Now a list of tensors
        "total_steps": total_steps,
    }


def load_examples_from_json(json_files, max_samples=None, seed=42):
    """
    Load examples from JSON files with option to limit to a random subset.

    Args:
        json_files: List of JSON file paths
        max_samples: Maximum number of examples to use (random selection)
        seed: Random seed for reproducibility

    Returns:
        List of valid examples
    """
    all_examples = []
    for file_path in json_files:
        logger.info(f"Loading examples from {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Check if the data is in the expected format
                if isinstance(data, dict) and (
                    "correct_examples" in data or "incorrect_examples" in data
                ):
                    correct_examples = data.get("correct_examples", [])
                    incorrect_examples = data.get("incorrect_examples", [])
                    logger.info(f"  Found {len(correct_examples)} correct examples")
                    logger.info(f"  Found {len(incorrect_examples)} incorrect examples")
                    all_examples.extend(correct_examples)
                    all_examples.extend(incorrect_examples)
                elif isinstance(data, list):
                    # Assume the data is a list of examples
                    logger.info(f"  Found {len(data)} examples in list format")
                    all_examples.extend(data)
                else:
                    # Try to infer the format
                    logger.warning(
                        f"Unrecognized data format in {file_path}, trying to infer structure..."
                    )
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                logger.info(
                                    f"  Adding {len(value)} examples from key '{key}'"
                                )
                                all_examples.extend(value)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue

    logger.info(f"Loaded {len(all_examples)} examples in total")

    # Filter out examples without required fields
    valid_examples = []
    for example in all_examples:
        if all(key in example for key in ["question", "response_steps", "step_value"]):
            valid_examples.append(example)

    logger.info(f"Found {len(valid_examples)} valid examples with all required fields")

    # Select a random subset if max_samples is specified
    if max_samples is not None and max_samples < len(valid_examples):
        import random

        random.seed(seed)  # Set seed for reproducibility
        valid_examples = random.sample(valid_examples, max_samples)
        logger.info(f"Randomly selected {max_samples} examples from the pool")

    return valid_examples


def save_checkpoint(
    model, optimizer, scheduler, epoch, loss, path, best_val_loss=None, global_step=None
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "global_step": global_step,
    }
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    experiment_name,
    output_dir,
    resume_from=None,
    log_steps=10,
    eval_steps=100,
    save_steps=1000,
    gradient_accumulation_steps=1,
    use_wandb=True,
):
    """
    Train the value head.

    Args:
        model: The model with value head
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer to use
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train for
        experiment_name: Name of the experiment
        output_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        log_steps: Steps between logging
        eval_steps: Steps between running evaluation
        save_steps: Steps between saving checkpoints
        gradient_accumulation_steps: Steps to accumulate gradients
        use_wandb: Whether to use wandb for logging
    """
    # Initialize wandb if needed
    if use_wandb:
        wandb.init(
            project="math_value_model",
            name=experiment_name,
            resume=True if resume_from else False,
        )
        wandb.config.update(
            {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                "model": model.__class__.__name__,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "loss_type": "MSE Loss",
                "log_steps": log_steps,
                "eval_steps": eval_steps,
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"best_{experiment_name}.pt")
    last_checkpoint_path = os.path.join(output_dir, f"checkpoint_{experiment_name}.pt")

    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(
            f"Resumed from checkpoint at epoch {start_epoch}, global step {global_step}"
        )

    # Create iterator for validation data
    val_iterator = iter(val_dataloader)

    try:
        for epoch in range(start_epoch, num_epochs):
            # Training loop
            model.train()
            train_loss = 0.0
            epoch_step = 0
            train_pbar = tqdm(
                train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
            )

            for batch_idx, batch in enumerate(train_pbar):
                if batch_idx % 100 == 0:
                    # Periodically clear cache to avoid memory issues
                    torch.cuda.empty_cache()
                    gc.collect()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Use target_values from the collated batch
                target_values = batch["target_values"]  # Already a list of tensors

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_values,
                )

                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    epoch_step += 1

                # Update metrics
                train_loss += loss.item() * gradient_accumulation_steps

                # Update progress bar
                current_lr = optimizer.param_groups[0]["lr"]
                train_pbar.set_postfix(
                    {
                        "loss": loss.item() * gradient_accumulation_steps,
                        "lr": current_lr,
                    }
                )

                # Log metrics
                if global_step % log_steps == 0 and use_wandb:
                    batch_loss = loss.item() * gradient_accumulation_steps
                    avg_train_loss = (
                        train_loss / epoch_step if epoch_step > 0 else batch_loss
                    )

                    wandb.log(
                        {
                            "train/batch_loss": batch_loss,
                            "train/loss": avg_train_loss,
                            "train/learning_rate": current_lr,
                            "global_step": global_step,
                            "epoch": epoch + (batch_idx / len(train_dataloader)),
                        }
                    )

                # Run periodic validation
                if global_step % eval_steps == 0:
                    model.eval()
                    val_batch_losses = []

                    # Run validation on a fixed number of batches
                    num_val_batches = min(10, len(val_dataloader))

                    with torch.no_grad():
                        for _ in range(num_val_batches):
                            try:
                                val_batch = next(val_iterator)
                            except StopIteration:
                                val_iterator = iter(val_dataloader)
                                val_batch = next(val_iterator)

                            val_input_ids = val_batch["input_ids"].to(device)
                            val_attention_mask = val_batch["attention_mask"].to(device)
                            val_target_values = val_batch["target_values"]

                            # Forward pass
                            val_outputs = model(
                                input_ids=val_input_ids,
                                attention_mask=val_attention_mask,
                                labels=val_target_values,
                            )

                            val_loss = val_outputs["loss"].item()
                            val_batch_losses.append(val_loss)

                    # Calculate average validation loss
                    periodic_val_loss = sum(val_batch_losses) / len(val_batch_losses)

                    if use_wandb:
                        wandb.log(
                            {
                                "eval/loss": periodic_val_loss,
                                "global_step": global_step,
                                "epoch": epoch + (batch_idx / len(train_dataloader)),
                            }
                        )

                    logger.info(
                        f"Step {global_step}: Validation Loss: {periodic_val_loss:.6f}"
                    )

                    # Switch back to training mode
                    model.train()

                # Save checkpoint
                if global_step % save_steps == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=loss.item() * gradient_accumulation_steps,
                        path=last_checkpoint_path,
                        best_val_loss=best_val_loss,
                        global_step=global_step,
                    )

            # Calculate average losses for the epoch
            avg_train_loss = train_loss / epoch_step if epoch_step > 0 else 0

            # Full validation loop at the end of each epoch
            model.eval()
            val_loss = 0.0
            val_steps = 0
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target_values = batch["target_values"]

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=target_values,
                    )

                    loss = outputs["loss"]

                    # Update metrics
                    val_loss += loss.item()
                    val_steps += 1

                    # Update progress bar
                    val_pbar.set_postfix({"loss": loss.item()})

            # Calculate average validation loss
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

            # Log epoch metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )

            if use_wandb:
                wandb.log(
                    {
                        "train/epoch_loss": avg_train_loss,
                        "eval/epoch_loss": avg_val_loss,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    }
                )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_to_save = model.module if hasattr(model, "module") else model

                best_model_path = os.path.join(output_dir, f"best_{experiment_name}.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    f"Saved best model with validation loss: {best_val_loss:.6f}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "eval/best_loss": best_val_loss,
                            "global_step": global_step,
                            "epoch": epoch + 1,
                        }
                    )

            # Save checkpoint at the end of each epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=avg_val_loss,
                path=last_checkpoint_path,
                best_val_loss=best_val_loss,
                global_step=global_step,
            )

    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving checkpoint...")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=val_loss / max(1, len(val_dataloader)),
            path=last_checkpoint_path,
            best_val_loss=best_val_loss,
            global_step=global_step,
        )

    if use_wandb:
        wandb.finish()

    return best_model_path


def push_to_hub(model_path, hub_model_id, token=None):
    """
    Push the model to Hugging Face Hub.

    Args:
        model_path: Path to the saved model
        hub_model_id: ID for the model on HF Hub (username/model_name)
        token: HF API token for authentication
    """
    logger.info(f"Pushing model to Hugging Face Hub: {hub_model_id}")

    if token:
        login(token=token)

    api = HfApi()
    api.create_repo(repo_id=hub_model_id, exist_ok=True)
    api.upload_folder(
        folder_path=model_path,
        repo_id=hub_model_id,
        commit_message="Upload mathematical value model",
    )

    logger.info(f"Model successfully pushed to {hub_model_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a value head on top of Qwen2.5-Math-PRM-7B"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="Base model name",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to JSON data files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="math_value_model",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100, help="Number of warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps between validation evaluations",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of examples to use (randomly selected)",
    )

    # Hugging Face Hub options
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID for Hugging Face Hub (username/model_name)",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hugging Face Hub token for authentication",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data with option to limit examples
    examples = load_examples_from_json(
        args.data_files, max_samples=args.max_samples, seed=args.seed
    )
    if not examples:
        logger.error("No valid examples found in the data files.")
        return

    # Split data into train and validation sets
    train_examples, val_examples = train_test_split(
        examples, test_size=0.1, random_state=args.seed
    )
    logger.info(
        f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}"
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    # Create datasets
    logger.info("Creating datasets")
    train_dataset = MathValueDataset(
        train_examples, tokenizer, max_length=args.max_length
    )
    val_dataset = MathValueDataset(val_examples, tokenizer, max_length=args.max_length)

    # Enhanced logging about sample counts
    logger.info(
        f"Created {len(train_dataset)} training samples (from {len(train_examples)} examples)"
    )
    logger.info(
        f"Created {len(val_dataset)} validation samples (from {len(val_examples)} examples)"
    )
    logger.info(
        f"Total samples: {len(train_dataset) + len(val_dataset)} (from {len(examples)} examples)"
    )
    logger.info(
        f"Average samples per example: {(len(train_dataset) + len(val_dataset)) / len(examples):.2f}"
    )

    # Create dataloaders with custom collation function
    num_workers = min(4, os.cpu_count() or 1)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use our custom collation function
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,  # Use our custom collation function
    )

    # Set up logging level for model debugging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("model").setLevel(logging.INFO)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    logger.info(f"Using compute dtype: {compute_dtype}")

    # First, load the config to check model type
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    logger.info(f"Loaded config: {config.__class__.__name__}")

    # Load the base PRM model with quantization
    logger.info(f"Loading base model {args.model_name} with quantization")
    try:
        # Try to load as a regular transformer model
        prm_model = AutoModel.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning(f"Error loading model with AutoModel: {e}")
        logger.info("Attempting to load with more specific model class")

        # Try a different approach based on the specific model architecture
        if "qwen" in args.model_name.lower():
            from transformers import Qwen2Model, Qwen2ForCausalLM

            try:
                prm_model = Qwen2ForCausalLM.from_pretrained(
                    args.model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as e2:
                logger.warning(f"Failed to load as Qwen2ForCausalLM: {e2}")
                prm_model = Qwen2Model.from_pretrained(
                    args.model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            raise ValueError(f"Could not load model: {e}")

    # Create our model with value head
    logger.info("Creating Qwen2PrmWithValueHead model")
    model = Qwen2PrmWithValueHead(prm_model)

    step_sep_token = "<extra_0>"
    step_sep_token_id = tokenizer.encode(step_sep_token)[0]
    model.step_sep_token_id = step_sep_token_id
    logger.info(f"Set step separator token ID to {step_sep_token_id}")

    # Print model structure for debugging
    logger.info(f"Model structure: {model.__class__.__name__}")
    logger.info(f"PRM model structure: {model.prm_model.__class__.__name__}")
    logger.info(f"Value head structure: {model.value_head}")

    # Print hidden size for debugging
    hidden_size = model.config.hidden_size
    logger.info(f"Model hidden size: {hidden_size}")

    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total {total_params:,})"
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.value_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Create learning rate scheduler
    total_steps = (
        len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Train model
    logger.info("Starting training")
    best_model_path = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        log_steps=10,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=not args.no_wandb,
    )

    logger.info(f"Training complete. Best adapter saved to {best_model_path}")

    final_model_dir = os.path.join(args.output_dir, args.experiment_name + "_trained")
    logger.info(f"Saving final model to {final_model_dir}")
    model.save_pretrained(final_model_dir)

    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        if not args.hub_model_id:
            logger.warning(
                "No hub_model_id provided, using experiment name as model ID"
            )
            hub_model_id = args.experiment_name
        else:
            hub_model_id = args.hub_model_id

        push_to_hub(
            model_path=final_model_dir, hub_model_id=hub_model_id, token=args.hub_token
        )


if __name__ == "__main__":
    main()
