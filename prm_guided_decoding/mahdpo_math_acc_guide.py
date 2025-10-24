"""
Value-guided decoding using ensembled multi-head DPO model as the base generator.
Combines the multi-head DPO inference approach with value model guidance.
"""

import os

# Clean up DDP environment variables
for v in ("LOCAL_RANK", "WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(v, None)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
)
from typing import List, Tuple, Optional, Dict
import json
from datasets import load_dataset
import gc
import argparse
import random
import logging
from tqdm import tqdm
import time

# Import the MultiHeadCausalLM model
from multihead_model import MultiHeadCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Keep the original Qwen2PrmWithValueHead class
class Qwen2PrmWithValueHead(PreTrainedModel):
    """
    Qwen2PrmWithValueHead adds a value prediction head on top of the Qwen2.5-Math-PRM-7B model.
    """

    def __init__(self, prm_model):
        super().__init__(prm_model.config)
        self.prm_model = prm_model
        self.step_sep_token = "<extra_0>"
        self.step_sep_token_id = None  # Will be set later

        # Freeze the base model parameters
        for param in self.prm_model.parameters():
            param.requires_grad = False

        hidden_size = self.config.hidden_size

        # Value head outputs unbounded values (no activation) for soft labels
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self._init_weights(self.value_head)

        # Move value head to the same device as the base model
        device = next(prm_model.parameters()).device
        self.value_head = self.value_head.to(device=device, dtype=torch.float32)
        logger.info(f"Value head moved to device: {device} with dtype: torch.float32")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Get model device from parameters
        model_device = next(self.parameters()).device

        # Ensure input_ids and attention_mask are on the right device
        if input_ids is not None and input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if attention_mask is not None and attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)

        outputs = self.prm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get hidden states from last layer
        hidden_states = outputs.hidden_states[-1]

        # Convert hidden states to the same dtype as the value head
        value_head_dtype = next(self.value_head.parameters()).dtype
        hidden_states = hidden_states.to(dtype=value_head_dtype)

        # Get batch size
        batch_size = hidden_states.shape[0]

        # Extract representations at ALL <extra_0> tokens
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
                if attention_mask is not None:
                    last_token_idx = min(
                        attention_mask[i].sum().item() - 1, hidden_states.shape[1] - 1
                    )
                else:
                    last_token_idx = hidden_states.shape[1] - 1
                last_hidden = hidden_states[i, last_token_idx].unsqueeze(0)
                step_values = self.value_head(last_hidden)
                all_step_values.append(step_values)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()

            # Match predictions with labels for each example in batch
            batch_losses = []
            for i, (step_values, label) in enumerate(zip(all_step_values, labels)):
                if hasattr(label, "shape") and len(label.shape) > 0:
                    # If label is a sequence, match length with predictions
                    pred_len = step_values.shape[0]
                    label_len = label.shape[0]

                    # Trim to shorter length
                    min_len = min(pred_len, label_len)
                    if min_len > 0:
                        # Calculate MSE loss for this example
                        example_loss = loss_fct(
                            step_values[:min_len].squeeze(-1),
                            label[:min_len].to(device=step_values.device),
                        )
                        batch_losses.append(example_loss)
                else:
                    # Single value label case
                    if step_values.shape[0] > 0:
                        example_loss = loss_fct(
                            step_values[-1].squeeze(-1),  # Use last step value
                            label.to(device=step_values.device),
                        )
                        batch_losses.append(example_loss)

            # Average losses across batch
            if batch_losses:
                loss = torch.stack(batch_losses).mean()

        return {
            "loss": loss,
            "value": all_step_values,  # Now returns list of tensors with values for all steps
            "prm_outputs": outputs,
        }

    def get_value(self, input_ids, attention_mask=None):
        """Get the value prediction for the given input."""
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["value"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        prm_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model = cls(prm_model)

        try:
            if "tokenizer" in kwargs:
                tokenizer = kwargs["tokenizer"]
                model.step_sep_token_id = tokenizer.encode("<extra_0>")[0]
        except Exception as e:
            logger.warning(f"Could not set step_sep_token_id from tokenizer: {e}")

        value_head_path = kwargs.get("value_head_path", None)
        if value_head_path is None:
            value_head_path = os.path.join(
                pretrained_model_name_or_path, "value_head.pt"
            )

        if os.path.exists(value_head_path):
            try:
                logger.info(f"Loading value head weights from {value_head_path}")
                device = next(model.parameters()).device
                value_head_state_dict = torch.load(value_head_path, map_location=device)
                model.value_head.load_state_dict(value_head_state_dict)
                logger.info(f"Successfully loaded value head weights")
            except Exception as e:
                logger.error(f"Error loading value head weights: {e}")
        else:
            try:
                from huggingface_hub import hf_hub_download

                value_head_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, filename="value_head.pt"
                )
                device = next(model.parameters()).device
                value_head_state_dict = torch.load(value_head_file, map_location=device)
                model.value_head.load_state_dict(value_head_state_dict)
                logger.info(
                    f"Successfully loaded value head weights from Hugging Face Hub"
                )
            except Exception as e:
                logger.warning(f"Could not load value head weights from HF Hub: {e}")
                logger.info(f"Value head will use default initialization")

        return model


def download_from_s3(s3_path: str, local_cache_dir: Optional[str] = None) -> str:
    """Download a file from S3 to a local cache directory."""
    import boto3
    import tempfile
    from pathlib import Path

    if not s3_path.startswith("s3://"):
        return s3_path

    s3_path = s3_path.replace("s3://", "")
    bucket_name, *key_parts = s3_path.split("/")
    key = "/".join(key_parts)

    if local_cache_dir is None:
        local_cache_dir = os.path.join(tempfile.gettempdir(), "model_cache")

    os.makedirs(local_cache_dir, exist_ok=True)

    key_path = Path(key)
    local_path = os.path.join(local_cache_dir, key_path.name)

    if not os.path.exists(local_path):
        logger.info(f"Downloading {s3_path} to {local_path}")
        s3_client = boto3.client("s3")
        try:
            s3_client.download_file(bucket_name, key, local_path)
            logger.info(f"Successfully downloaded file from S3")
        except Exception as e:
            logger.error(f"Error downloading file from S3: {e}")
            raise
    else:
        logger.info(f"Using cached file at {local_path}")

    return local_path


def download_from_hf_hub(
    hf_path: str, filename: str = "policy.pt", local_cache_dir: Optional[str] = None
) -> str:
    """Download a file from Hugging Face Hub to a local cache directory."""
    from huggingface_hub import hf_hub_download

    if hf_path.startswith("hf://"):
        path_parts = hf_path[5:].split("/")
        if len(path_parts) >= 3:
            repo_id = "/".join(path_parts[:2])
            filename = "/".join(path_parts[2:])
        else:
            repo_id = "/".join(path_parts)
    else:
        repo_id = hf_path

    logger.info(f"Downloading from Hugging Face Hub: {repo_id}/{filename}")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=local_cache_dir
        )
        logger.info(f"Successfully downloaded from HF Hub to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Error downloading file from HF Hub: {e}")
        raise


def is_hf_hub_path(path: str) -> bool:
    """Check if a path is a Hugging Face Hub path."""
    if path.startswith("hf://"):
        return True

    if "/" in path and not path.startswith("/") and not path.startswith("./"):
        parts = path.split("/")
        if len(parts) == 2 and not any(
            part.endswith((".pt", ".pth", ".bin")) for part in parts
        ):
            return True
        if (
            len(parts) >= 2
            and not path.startswith("s3://")
            and not os.path.exists(path)
        ):
            return True

    return False


def _clone_structure(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_structure(v) for v in x)
    if isinstance(x, dict):
        return {k: _clone_structure(v) for k, v in x.items()}
    if hasattr(x, "__dict__"):
        y = x.__class__.__new__(x.__class__)
        for k, v in x.__dict__.items():
            setattr(y, k, _clone_structure(v))
        return y
    try:
        import copy as _cpy

        return _cpy.copy(x)
    except Exception:
        return x


def clone_past_kv(past_kv):
    import copy as _cpy

    try:
        return _cpy.deepcopy(past_kv)
    except Exception:
        return _clone_structure(past_kv)


from dataclasses import dataclass


@dataclass
class RunningState:
    past_kv: object
    logits: torch.Tensor  # shape [1, vocab]


@dataclass
class Candidate:
    text: str
    token_ids: List[int]
    is_eos: bool
    finish_reason: str
    past_kv: object
    logits: torch.Tensor


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    generator: torch.Generator = None,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1))
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)

    if top_k and top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, topk_idx, topk_vals)
        probs = filtered
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_id = torch.multinomial(probs.squeeze(0), num_samples=1, generator=generator)
    return int(next_id.item())


class DPOEnsembleValueGuidedDecoder:
    """
    Value-guided decoder that uses an ensembled multi-head DPO model as the base generator.
    Combines the power of multi-head DPO with value model guidance.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dpo_checkpoint_path: str = None,
        value_model_name: str = "Jennny/qwen-math-value-model-join",
        reward_model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        num_heads: int = 2,
        max_steps: int = 20,
        num_candidates: int = 5,
        base_device_id: int = 0,  # GPU for DPO model
        value_device_id: int = 1,  # GPU for value/reward model
        dtype: str = "bfloat16",
        cache_dir: Optional[str] = None,
        use_value_guidance: bool = True,
        use_prm_guidance: bool = False,
        use_base_model_only: bool = False,
    ):
        self.use_value_guidance = use_value_guidance
        self.use_prm_guidance = use_prm_guidance
        self.use_base_model_only = use_base_model_only

        # Set devices
        self.base_device = torch.device(f"cuda:{base_device_id}")
        self.value_device = torch.device(f"cuda:{value_device_id}")

        if self.use_base_model_only:
            if self.use_value_guidance:
                logger.info(f"Base Model + Value Guidance Setup:")
                logger.info(f"   Base model -> GPU {base_device_id}")
                logger.info(f"   Value model -> GPU {value_device_id}")
            elif self.use_prm_guidance:
                logger.info(f"Base Model + PRM Guidance Setup:")
                logger.info(f"   Base model -> GPU {base_device_id}")
                logger.info(f"   PRM (reward) model -> GPU {value_device_id}")
            else:
                logger.info(f"Base Model Only Setup:")
                logger.info(f"   Base model -> GPU {base_device_id}")
        else:
            logger.info(f"Dual-GPU Setup (DPO Ensemble + Value Model):")
            logger.info(f"   DPO Ensemble model -> GPU {base_device_id}")
            if self.use_prm_guidance:
                logger.info(f"   PRM (reward) model -> GPU {value_device_id}")
            else:
                logger.info(f"   Value model -> GPU {value_device_id}")

        # Set torch dtype
        self.torch_dtype = getattr(torch, dtype)

        # Load tokenizer
        logger.info(f"Loading tokenizer: {base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.base_tokenizer.pad_token_id is None:
            self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

        # *** KEY CHANGE: Conditional model loading ***
        if self.use_base_model_only:
            # Load base model directly without MultiHeadCausalLM wrapper
            logger.info(f"Loading base model directly: {base_model_name}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map={"": self.base_device},
            ).eval()
            self.dpo_model = None  # No DPO model needed
        else:
            # Original DPO ensemble path
            logger.info(f"Loading base model for DPO: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )

            # Create MultiHeadCausalLM model
            logger.info(f"Creating multi-head DPO model with {num_heads} heads")
            self.dpo_model = MultiHeadCausalLM(base_model, num_heads=num_heads).to(
                self.base_device
            )
            self.base_model = None  # Not needed when using DPO

            # Load DPO checkpoint
            if dpo_checkpoint_path:
                local_checkpoint_path = None

                if dpo_checkpoint_path.startswith("s3://"):
                    if not dpo_checkpoint_path.endswith(".pt"):
                        if not dpo_checkpoint_path.endswith("/"):
                            dpo_checkpoint_path += "/"
                        dpo_checkpoint_path += "policy.pt"
                    local_checkpoint_path = download_from_s3(
                        dpo_checkpoint_path, cache_dir
                    )

                elif is_hf_hub_path(dpo_checkpoint_path):
                    local_checkpoint_path = download_from_hf_hub(
                        dpo_checkpoint_path,
                        filename="policy.pt",
                        local_cache_dir=cache_dir,
                    )
                else:
                    if os.path.isdir(dpo_checkpoint_path):
                        local_checkpoint_path = os.path.join(
                            dpo_checkpoint_path, "policy.pt"
                        )
                    else:
                        local_checkpoint_path = dpo_checkpoint_path

                logger.info(f"Loading DPO checkpoint: {local_checkpoint_path}")
                checkpoint = torch.load(local_checkpoint_path, map_location="cpu")

                if "state" in checkpoint:
                    state_dict = checkpoint["state"]
                    self.dpo_model.load_state_dict(state_dict, strict=False)
                else:
                    self.dpo_model.load_state_dict(checkpoint, strict=False)

                logger.info("Successfully loaded DPO checkpoint")

            self.dpo_model.eval()

        # Value/PRM model loading when guidance is enabled
        if self.use_value_guidance:
            logger.info(f"Loading value model on GPU {value_device_id}...")
            self.value_model = Qwen2PrmWithValueHead.from_pretrained(
                value_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self.value_device},
                trust_remote_code=True,
            ).eval()
            self.value_tokenizer = AutoTokenizer.from_pretrained(
                value_model_name, trust_remote_code=True
            )
            self.step_token = "<extra_0>"
            step_sep_token_id = self.value_tokenizer.encode(self.step_token)[0]
            self.value_model.step_sep_token_id = step_sep_token_id
        else:
            self.value_model = None
            self.value_tokenizer = None
            self.step_token = "<extra_0>"

        # PRM (reward) model loading when enabled
        if self.use_prm_guidance:
            logger.info(f"Loading PRM (reward) model on GPU {value_device_id}...")
            self.reward_model = AutoModel.from_pretrained(
                reward_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self.value_device},
                trust_remote_code=True,
            ).eval()
            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                reward_model_name, trust_remote_code=True
            )
            # Ensure step token known for PRM path too
            self.step_token = "<extra_0>"
        else:
            self.reward_model = None
            self.reward_tokenizer = None

        self.max_steps = max_steps
        self.num_candidates = num_candidates
        self.num_heads = num_heads

        self.system_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        if self.use_base_model_only:
            if self.use_value_guidance:
                logger.info(f"Base Model + Value Guidance setup complete!")
            elif self.use_prm_guidance:
                logger.info(f"Base Model + PRM Guidance setup complete!")
            else:
                logger.info(f"Base Model Only setup complete!")
        else:
            if self.use_prm_guidance:
                logger.info(f"DPO Ensemble + PRM Guidance setup complete!")
            else:
                logger.info(f"DPO Ensemble + Value Model setup complete!")

    def build_conversation_prompt(self, query: str, previous_steps: List[str]) -> str:
        """Build conversation prompt properly to avoid multiple assistant messages."""
        if not previous_steps:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ]
            prompt = self.base_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            current_text = "\n\n".join(previous_steps)
            current_text += "\n\n"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": current_text},
            ]
            prompt = self.base_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        return prompt

    def extract_next_step(self, generated_text: str, input_prompt: str) -> str:
        """Extract the next step from generated text."""
        if generated_text.startswith(input_prompt):
            new_content = generated_text[len(input_prompt) :]
        else:
            new_content = generated_text

        return new_content.strip()

    def extract_assistant_response(self, full_text: str) -> str:
        """Extract only the assistant's response from the full generated text."""
        # First try to remove any system prompt
        if "system\n" in full_text:
            full_text = full_text.split("system\n", 1)[1]

        # Try to find content after the last "Assistant:" or "assistant" tag
        import re

        assistant_pattern = re.compile(
            r"(?:^|\n)(?:Assistant:|assistant\n)(.*?)(?:$|\n\n(?:Human:|Human\n|user\n|$))",
            re.DOTALL | re.IGNORECASE,
        )
        matches = assistant_pattern.findall(full_text)

        if matches:
            # Return the last match (most recent assistant response)
            return matches[-1].strip()

        # If we didn't find a clear assistant marker, try splitting on 'user'
        if "user\n" in full_text:
            parts = full_text.split("user\n", 1)[1]
            if "assistant\n" in parts:
                return parts.split("assistant\n", 1)[1].strip()

        # Last resort - just return the whole text
        return full_text.strip()

    def calculate_value(
        self, steps: List[str], query: str
    ) -> Tuple[float, List[float]]:
        """Calculate values using the value model on value_device."""
        if not steps:
            return 0.0, []

        formatted_response = f"{self.step_token}".join(steps) + self.step_token
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": formatted_response},
        ]

        conversation = self.value_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        input_ids = self.value_tokenizer.encode(conversation, return_tensors="pt").to(
            self.value_device
        )
        attention_mask = torch.ones_like(input_ids).to(self.value_device)

        with torch.no_grad():
            all_values = self.value_model.get_value(input_ids, attention_mask)
            step_values = all_values[0].squeeze(-1).cpu().tolist()

            if len(step_values) > len(steps):
                step_values = step_values[: len(steps)]
            elif len(step_values) < len(steps):
                step_values = step_values + [0.0] * (len(steps) - len(step_values))

            cumulative_value = sum(step_values)

        del input_ids, attention_mask, all_values
        torch.cuda.empty_cache()

        return cumulative_value, step_values

    def _process_prm_rewards(
        self, logits: torch.Tensor, token_masks: torch.Tensor
    ) -> List[List[float]]:
        """Process PRM logits at step separators to per-step reward probabilities.

        Expects logits of shape [batch, seq_len, 2] where index 1 corresponds to positive reward.
        """
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        rewards: List[List[float]] = []
        for sample_probs in probabilities:  # [seq_len, 2]
            masked = sample_probs[sample_probs.sum(dim=-1) != 0]  # keep step positions
            if masked.numel() == 0:
                rewards.append([])
            else:
                # Take the probability of the positive class (index 1)
                rewards.append(masked[:, 1].detach().cpu().tolist())
        return rewards

    def calculate_reward(
        self, steps: List[str], query: str
    ) -> Tuple[float, List[float]]:
        """Calculate PRM rewards on value_device using the reward model (PRM)."""
        if not steps or self.reward_model is None or self.reward_tokenizer is None:
            return 0.0, []

        formatted_response = f"{self.step_token}".join(steps) + self.step_token
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": formatted_response},
        ]

        conversation = self.reward_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        input_ids = self.reward_tokenizer.encode(conversation, return_tensors="pt").to(
            self.value_device
        )

        with torch.no_grad():
            outputs = self.reward_model(input_ids=input_ids)

        step_sep_id = self.reward_tokenizer.encode(self.step_token)[0]
        token_masks = (input_ids == step_sep_id).to(
            dtype=outputs[0].dtype, device=outputs[0].device
        )
        rewards_list = self._process_prm_rewards(outputs[0], token_masks)

        if rewards_list and len(rewards_list[0]) > 0:
            step_rewards = rewards_list[0]
            cumulative_reward = float(sum(step_rewards))
        else:
            step_rewards = []
            cumulative_reward = 0.0

        del input_ids, outputs, token_masks
        torch.cuda.empty_cache()

        return cumulative_reward, step_rewards

    def generate_candidate_steps_base_only(
        self,
        query: str,
        previous_steps: List[str],
        num_candidates: Optional[int] = None,
    ) -> List[str]:
        """Generate candidate steps using the base model directly."""
        # Use provided num_candidates or default to instance setting
        target_candidates = (
            num_candidates if num_candidates is not None else self.num_candidates
        )

        logger.info(
            f"🔧 Using FULL REGENERATION method to generate {target_candidates} candidate(s)"
        )
        logger.info(f"   - Will rebuild full prompt + context for each candidate")
        logger.info(f"   - No hidden state caching (standard HF generate)")

        formatted_prompt = self.build_conversation_prompt(query, previous_steps)

        # Tokenize
        inputs = self.base_tokenizer(
            formatted_prompt, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.base_device) for k, v in inputs.items()}

        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": self.base_tokenizer.pad_token_id,
            "eos_token_id": self.base_tokenizer.eos_token_id,
        }

        candidates = []

        # Generate multiple candidates using base model
        for i in range(target_candidates):
            logger.info(
                f"   🔄 Generating candidate #{i+1} with full prompt regeneration"
            )
            try:
                with torch.no_grad():
                    # Use base model generation directly
                    outputs = self.base_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_kwargs,
                    )

                    full_text = self.base_tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )

                    # Extract the new content (same logic as existing methods)
                    response = self.extract_assistant_response(full_text)

                    # Find the new step after the previous steps
                    if previous_steps:
                        combined_previous = "\n\n".join(previous_steps)
                        if response.startswith(combined_previous):
                            new_step = response[len(combined_previous) :].strip()
                            # Remove leading \n\n if present
                            if new_step.startswith("\n\n"):
                                new_step = new_step[2:]
                        else:
                            # Try to find where the new content starts
                            new_step = response
                            for prev_step in previous_steps:
                                if new_step.startswith(prev_step):
                                    new_step = new_step[len(prev_step) :].strip()
                    else:
                        new_step = response

                    # Clean up the step
                    new_step = new_step.strip()

                    # Stop at double newline
                    if "\n\n" in new_step:
                        new_step = new_step.split("\n\n")[0].strip()

                    if self._is_valid_step(new_step):
                        candidates.append(new_step)
                        logger.info(
                            f"      Valid candidate: {new_step[:50]}{'...' if len(new_step) > 50 else ''}"
                        )
                    else:
                        logger.info(f"      Invalid candidate")

            except Exception as e:
                logger.warning(f"      Error generating candidate {i+1}: {e}")
                continue

        # Clean up
        del inputs
        if "outputs" in locals():
            del outputs
        torch.cuda.empty_cache()

        logger.info(
            f"Full regeneration method completed: generated {len(candidates)} candidate(s)"
        )
        return candidates

    def generate_candidate_steps_hidden(
        self,
        query: str,
        previous_steps: List[str],
        num_candidates: Optional[int] = None,
    ) -> List[str]:
        """Generate candidate steps using one prompt pass and cached hidden state."""
        # Use provided num_candidates or default to instance setting
        target_candidates = (
            num_candidates if num_candidates is not None else self.num_candidates
        )

        logger.info(
            f"🔧 Using HIDDEN STATE CACHING method to generate {target_candidates} candidate(s)"
        )
        logger.info(f"   - Will cache past_key_values after initial forward pass")
        logger.info(
            f"   - Will do incremental token-by-token generation for each candidate"
        )

        # Build prompt once
        prompt = self.build_conversation_prompt(query, previous_steps)
        enc = self.base_tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.base_device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.base_device)

        # Seed for reproducibility compatible with your runner
        import time as _t

        base_seed = int((_t.time_ns() % 2_147_483_647))
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)

        # Choose which model to use based on mode
        if self.use_base_model_only:
            model = self.base_model
        else:
            model = self.dpo_model

        # Initial forward to get cache and next token logits
        with torch.no_grad():
            if self.use_base_model_only:
                # Base model forward pass
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # DPO ensemble forward pass with head_index=None
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    head_index=None,
                )
        main_past = out.past_key_values
        main_logits = out.logits[:, -1, :]

        prompt_tokens = input_ids.shape[1]
        logger.info(f"   Cached past_key_values for {prompt_tokens} prompt tokens")
        logger.info(f"   Starting incremental generation from cached state...")

        # Local sampling settings, same as your generate path
        temperature = 1.0
        top_p = 1.0
        top_k = 50
        max_new = 512
        min_chars_before_boundary = (
            1  # keep very permissive, you already trim at the first blank line
        )

        candidates: List[str] = []
        attempts = 0
        max_attempts = max(target_candidates, 1) * 3

        # Inner sampler for one candidate
        def sample_one(start_past, start_logits) -> Tuple[str, tuple, torch.Tensor]:
            local_past = clone_past_kv(start_past)
            local_logits = start_logits.detach().clone()
            picked: List[int] = []
            device = local_logits.device
            gen = torch.Generator(device=device)
            gen.manual_seed(base_seed + len(candidates) * 10007 + attempts)

            # Stream tokens until we hit a blank line boundary or quota
            new_text = ""
            tokens_emitted = 0
            eos_id = self.base_tokenizer.eos_token_id

            while tokens_emitted < max_new:
                nid = _sample_next_token(
                    local_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=gen,
                )
                picked.append(nid)
                tokens_emitted += 1

                # EOS check
                if eos_id is not None and nid == eos_id:
                    break

                # Advance one step - handle both model types
                with torch.no_grad():
                    if self.use_base_model_only:
                        # Base model step
                        out_step = model(
                            input_ids=torch.tensor(
                                [[nid]], device=device, dtype=torch.long
                            ),
                            past_key_values=local_past,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        # DPO ensemble step
                        out_step = model(
                            input_ids=torch.tensor(
                                [[nid]], device=device, dtype=torch.long
                            ),
                            past_key_values=local_past,
                            use_cache=True,
                            return_dict=True,
                            head_index=None,
                        )
                local_past = out_step.past_key_values
                local_logits = out_step.logits[:, -1, :]

                # Decode incrementally and check boundary
                new_text = self.base_tokenizer.decode(picked, skip_special_tokens=True)

                # Keep your original boundary rule, stop at first double newline
                if len(new_text) >= min_chars_before_boundary and "\n\n" in new_text:
                    # Trim to the first chunk before the blank line
                    new_text = new_text.split("\n\n", 1)[0].strip()
                    break

            return new_text.strip(), local_past, local_logits

        # Collect up to target_candidates steps
        while len(candidates) < target_candidates and attempts < max_attempts:
            attempts += 1
            logger.info(
                f"   🎲 Generating candidate #{len(candidates)+1} using cached hidden state (attempt {attempts})"
            )
            cand_text, cand_past, cand_logits = sample_one(main_past, main_logits)
            if not self._is_valid_step(cand_text):
                logger.info(f"      Invalid candidate, retrying...")
                continue
            candidates.append(cand_text)
            logger.info(
                f"      Valid candidate: {cand_text[:50]}{'...' if len(cand_text) > 50 else ''}"
            )

        # Cleanup big tensors
        del out, main_past, main_logits, enc
        torch.cuda.empty_cache()

        logger.info(
            f"Hidden state caching method completed: generated {len(candidates)} candidate(s) in {attempts} attempts"
        )
        return candidates

    def evaluate_candidates(
        self, candidates: List["Candidate"], query: str, previous_steps: List[str]
    ) -> List[Tuple["Candidate", float, List[float]]]:
        """Evaluate candidates with the value model. Returns (Candidate, immediate_value, per_step_values)."""
        results: List[Tuple[Candidate, float, List[float]]] = []
        current_cumulative_value, current_step_values = (
            self.calculate_value(previous_steps, query) if previous_steps else (0.0, [])
        )

        for cand in candidates:
            if not self._is_valid_step(cand.text):
                continue
            test_steps = previous_steps + [cand.text]
            new_cumulative_value, new_step_values = self.calculate_value(
                test_steps, query
            )
            if len(new_step_values) > len(previous_steps):
                immediate_value = new_step_values[len(previous_steps)]
            else:
                immediate_value = new_cumulative_value - current_cumulative_value

            results.append((cand, immediate_value, new_step_values))
            del test_steps
            torch.cuda.empty_cache()

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def evaluate_candidates_prm(
        self, candidates: List["Candidate"], query: str, previous_steps: List[str]
    ) -> List[Tuple["Candidate", float, List[float]]]:
        """Evaluate candidates using PRM. Returns (Candidate, immediate_reward, per_step_rewards)."""
        results: List[Tuple[Candidate, float, List[float]]] = []
        for cand in candidates:
            if not self._is_valid_step(cand.text):
                continue
            test_steps = previous_steps + [cand.text]
            cumulative_reward, step_rewards = self.calculate_reward(test_steps, query)
            immediate_reward = step_rewards[-1] if step_rewards else 0.0
            results.append((cand, immediate_reward, step_rewards))
            del test_steps
            torch.cuda.empty_cache()
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def decode_base_with_value(self, query: str) -> Tuple[str, List[float], float]:
        """Generate solution using base model with value guidance."""
        steps = []
        all_step_values = []
        cumulative_value = 0.0

        logger.info(
            f"\nStarting Base Model + Value Guidance decoding for query:\n{query}\n"
        )

        for step_num in range(self.max_steps):
            if steps and "\\boxed{" in steps[-1]:
                logger.info(f"\nSolution complete with boxed answer")
                break

            if steps:
                total_tokens = len(self.base_tokenizer.encode("\n\n".join(steps)))
                if total_tokens > 4000:
                    logger.info(
                        f"\nWARNING: Solution reached {total_tokens} tokens, stopping early."
                    )
                    break

            logger.info(f"\n🔍 Generating step {step_num + 1}/{self.max_steps}")
            if steps:
                logger.info(
                    f"Current solution (Cumulative Value: {cumulative_value:.8f}):"
                )
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")
                    if i <= len(all_step_values):
                        logger.info(f"           (Value: {all_step_values[i-1]:.4f})")

            # Generate candidates using base model
            candidates = self.generate_candidate_steps_base_only(query, steps)
            if not candidates:
                logger.info(f"WARNING: No valid candidates generated, stopping.")
                break

            # Evaluate candidates using value model
            evaluated_candidates = self.evaluate_candidates(candidates, query, steps)
            if not evaluated_candidates:
                logger.info(f"WARNING: No valid candidates after evaluation, stopping.")
                break

            best_candidate, best_immediate_value, new_step_values = (
                evaluated_candidates[0]
            )

            logger.info("   Candidate immediate values:")
            for i, (cand, value, _) in enumerate(
                evaluated_candidates[: min(3, len(evaluated_candidates))], 1
            ):
                logger.info(f"   Candidate {i}: {cand[:50]}... (Value: {value:.4f})")

            steps.append(best_candidate)
            all_step_values = new_step_values
            cumulative_value = cumulative_value + best_immediate_value
            logger.info(f"   Selected: {best_candidate[:60]}...")
            logger.info(f"   Immediate Value: {best_immediate_value:.4f}")

            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\n💡 Final solution (Total Value: {cumulative_value:.8f}):")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")
            if i <= len(all_step_values):
                logger.info(f"           (Value: {all_step_values[i-1]:.4f})")

        return solution, all_step_values, cumulative_value

    def decode_base_hidden_with_value(
        self, query: str
    ) -> Tuple[str, List[float], float]:
        """Base model + value guidance, with true running KV cache across steps."""
        steps: List[str] = []
        all_step_values: List[float] = []
        cumulative_value: float = 0.0

        logger.info(
            f"\nStarting Base Model + Hidden State + Value Guidance decoding for query:\n{query}\n"
        )

        # Seed running state once
        state = self._initial_forward(query)

        for step_num in range(self.max_steps):
            if steps and "\\boxed{" in steps[-1]:
                logger.info(f"\nSolution complete with boxed answer")
                break

            if steps:
                total_tokens = len(self.base_tokenizer.encode("\n\n".join(steps)))
                if total_tokens > 4000:
                    logger.info(
                        f"\nWARNING: Solution reached {total_tokens} tokens, stopping early."
                    )
                    break

            logger.info(f"\n🔍 Generating step {step_num + 1}/{self.max_steps}")
            if steps:
                logger.info(
                    f"Current solution (Cumulative Value: {cumulative_value:.8f}):"
                )
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")
                    if i <= len(all_step_values):
                        logger.info(f"           (Value: {all_step_values[i-1]:.4f})")

            cands = self._generate_candidates_from_state(state)
            if not cands:
                logger.info(f"WARNING: No valid candidates generated, stopping.")
                break
            if len(cands) == 1 and cands[0].is_eos and not cands[0].text:
                logger.info("WARNING: EOS only, stopping.")
                break

            evaluated = self.evaluate_candidates(cands, query, steps)
            if not evaluated:
                logger.info(f"WARNING: No valid candidates after evaluation, stopping.")
                break

            best_cand, best_immediate, new_step_values = evaluated[0]
            logger.info("   Candidate immediate values:")
            for i, (c, v, _) in enumerate(evaluated[: min(3, len(evaluated))], 1):
                logger.info(f"   Candidate {i}: {c.text[:50]}... (Value: {v:.4f})")

            steps.append(best_cand.text)
            all_step_values = new_step_values
            cumulative_value += best_immediate
            logger.info(f"   Selected: {best_cand.text[:60]}...")
            logger.info(f"   Immediate Value: {best_immediate:.4f}")

            state = self._promote(best_cand)

            for c in cands:
                if c is not best_cand:
                    del c.past_kv, c.logits
            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\n💡 Final solution (Total Value: {cumulative_value:.8f}):")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")
            if i <= len(all_step_values):
                logger.info(f"           (Value: {all_step_values[i-1]:.4f})")

        return solution, all_step_values, cumulative_value

    def decode_base_hidden_with_prm(self, query: str) -> Tuple[str, List[float], float]:
        """Base model + PRM guidance, with true running KV cache across steps."""
        steps: List[str] = []
        all_step_rewards: List[float] = []
        cumulative_reward: float = 0.0

        logger.info(
            f"\nStarting Base Model + Hidden State + PRM Guidance decoding for query:\n{query}\n"
        )

        state = self._initial_forward(query)

        for step_num in range(self.max_steps):
            if steps and "\\boxed{" in steps[-1]:
                logger.info(f"\nSolution complete with boxed answer")
                break

            if steps:
                total_tokens = len(self.base_tokenizer.encode("\n\n".join(steps)))
                if total_tokens > 4000:
                    logger.info(
                        f"\nWARNING: Solution reached {total_tokens} tokens, stopping early."
                    )
                    break

            logger.info(f"\n🔍 Generating step {step_num + 1}/{self.max_steps}")
            if steps:
                logger.info(
                    f"Current solution (Cumulative Reward: {cumulative_reward:.8f}):"
                )
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")
                    if i <= len(all_step_rewards):
                        logger.info(f"           (Reward: {all_step_rewards[i-1]:.4f})")

            cands = self._generate_candidates_from_state(state)
            if not cands:
                logger.info(f"WARNING: No valid candidates generated, stopping.")
                break
            if len(cands) == 1 and cands[0].is_eos and not cands[0].text:
                logger.info("WARNING: EOS only, stopping.")
                break

            evaluated = self.evaluate_candidates_prm(cands, query, steps)
            if not evaluated:
                logger.info(
                    f"WARNING: No valid candidates after PRM evaluation, stopping."
                )
                break

            best_cand, best_immediate_reward, new_step_rewards = evaluated[0]
            logger.info("   Candidate immediate rewards:")
            for i, (c, v, _) in enumerate(evaluated[: min(3, len(evaluated))], 1):
                logger.info(f"   Candidate {i}: {c.text[:50]}... (Reward: {v:.4f})")

            steps.append(best_cand.text)
            all_step_rewards = new_step_rewards
            cumulative_reward += best_immediate_reward
            logger.info(f"   Selected: {best_cand.text[:60]}...")
            logger.info(f"   Immediate Reward: {best_immediate_reward:.4f}")

            state = self._promote(best_cand)

            for c in cands:
                if c is not best_cand:
                    del c.past_kv, c.logits
            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\n💡 Final solution (Total Reward: {cumulative_reward:.8f}):")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")
            if i <= len(all_step_rewards):
                logger.info(f"           (Reward: {all_step_rewards[i-1]:.4f})")

        return solution, all_step_rewards, cumulative_reward

    def decode_base_only(self, query: str) -> Tuple[str, List[float], float]:
        """Generate solution using base model only, no value guidance."""
        steps = []
        all_step_values = []  # Empty for compatibility
        cumulative_value = 0.0  # Zero for compatibility

        logger.info(f"\nStarting Base Model Only decoding for query:\n{query}\n")

        for step_num in range(self.max_steps):
            if steps and "\\boxed{" in steps[-1]:
                logger.info(f"\nSolution complete with boxed answer")
                break

            if steps:
                total_tokens = len(self.base_tokenizer.encode("\n\n".join(steps)))
                if total_tokens > 4000:
                    logger.info(
                        f"\nWARNING: Solution reached {total_tokens} tokens, stopping early."
                    )
                    break

            logger.info(f"\n🔍 Generating step {step_num + 1}/{self.max_steps}")
            if steps:
                logger.info(f"Current solution:")
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")

            # Generate candidates using base model
            # Generate only 1 candidate since no value guidance is used
            candidates = self.generate_candidate_steps_base_only(
                query, steps, num_candidates=1
            )
            if not candidates:
                logger.info(f"WARNING: No valid candidates generated, stopping.")
                break

            # Just take the first candidate (no value evaluation)
            best_candidate = candidates[0]
            steps.append(best_candidate)

            logger.info(f"   Selected: {best_candidate[:60]}...")

            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\n💡 Final solution:")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")

        return solution, all_step_values, cumulative_value

    def decode_base_hidden_no_value(self, query: str) -> Tuple[str, List[float], float]:
        """Base model with hidden state sampling, no value guidance. Carry KV across steps."""
        steps: List[str] = []
        all_step_values: List[float] = []
        cumulative_value: float = 0.0

        logger.info(
            f"\nStarting Base Model + Hidden State (No Value Guidance) decoding for query:\n{query}\n"
        )

        state = self._initial_forward(query)

        for step_num in range(self.max_steps):
            if steps and "\\boxed{" in steps[-1]:
                logger.info(f"\nSolution complete with boxed answer")
                break

            if steps:
                total_tokens = len(self.base_tokenizer.encode("\n\n".join(steps)))
                if total_tokens > 4000:
                    logger.info(
                        f"\nWARNING: Solution reached {total_tokens} tokens, stopping early."
                    )
                    break

            logger.info(f"\n🔍 Generating step {step_num + 1}/{self.max_steps}")
            if steps:
                logger.info(f"Current solution:")
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")

            cands = self._generate_candidates_from_state(state, num_candidates=1)
            if not cands:
                logger.info(f"WARNING: No valid candidates generated, stopping.")
                break

            best_cand = cands[0]
            if not self._is_valid_step(best_cand.text):
                logger.info(f"WARNING: Invalid candidate, stopping.")
                break

            steps.append(best_cand.text)
            logger.info(f"   Selected: {best_cand.text[:60]}...")

            state = self._promote(best_cand)

            for c in cands:
                if c is not best_cand:
                    del c.past_kv, c.logits
            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\n💡 Final solution:")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")

        return solution, all_step_values, cumulative_value

    def decode(self, query: str) -> Tuple[str, List[float], float]:
        """Alias to hidden-state value-guided DPO decode (cache-carrying)."""
        return self.decode_hidden(query)

    def decode_base_hidden(self, query: str) -> Tuple[str, List[float], float]:
        """Step by step decode using hidden state sampling, no value guidance."""
        steps: List[str] = []
        all_step_values: List[float] = []  # kept for API compatibility, stays empty
        cumulative_value: float = 0.0  # kept for API compatibility, stays zero

        logger.info(f"\nStarting base hidden state decoding for query:\n{query}\n")

        for step_num in range(self.max_steps):
            if steps and "\\boxed{" in steps[-1]:
                logger.info(f"\nSolution complete with boxed answer")
                break

            if steps:
                total_tokens = len(self.base_tokenizer.encode("\n\n".join(steps)))
                if total_tokens > 4000:
                    logger.info(
                        f"\nSolution reached {total_tokens} tokens, stopping early."
                    )
                    break

            logger.info(f"\nGenerating step {step_num + 1}/{self.max_steps}")
            if steps:
                logger.info(f"Current solution:")
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")

            # Use the unified hidden state sampler (works for both base and DPO models)
            # Generate only 1 candidate since no value guidance is used
            candidates = self.generate_candidate_steps_hidden(
                query, steps, num_candidates=1
            )
            if not candidates:
                logger.info(f"No valid candidates generated, stopping.")
                break

            # No value model, just take the first candidate
            chosen = candidates[0]
            logger.info(f"   Selected: {chosen[:60]}...")

            steps.append(chosen)

            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\nFinal solution:")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")

        # Return the same tuple shape as other decode methods
        return solution, all_step_values, cumulative_value

    def decode_hidden(self, query: str) -> Tuple[str, List[float], float]:
        """DPO ensemble + value model, with true running KV cache across steps."""
        steps: List[str] = []
        all_step_values: List[float] = []
        cumulative_value: float = 0.0

        logger.info(
            f"\nStarting DPO Ensemble Value guided decoding with hidden state for query:\n{query}\n"
        )

        # Seed running state once
        state = self._initial_forward(query)

        for step_num in range(self.max_steps):
            if steps and "\\boxed{" in steps[-1]:
                logger.info(f"\nSolution complete with boxed answer")
                break

            # conservative cap using your existing approximation
            if steps:
                total_tokens = len(self.base_tokenizer.encode("\n\n".join(steps)))
                if total_tokens > 4000:
                    logger.info(
                        f"\nSolution reached {total_tokens} tokens, stopping early."
                    )
                    break

            logger.info(f"\nGenerating step {step_num + 1}/{self.max_steps}")
            if steps:
                logger.info(
                    f"Current solution (Cumulative Value: {cumulative_value:.8f}):"
                )
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")
                    if i <= len(all_step_values):
                        logger.info(f"           (Value: {all_step_values[i-1]:.4f})")

            # propose from current state
            cands = self._generate_candidates_from_state(state)
            if not cands:
                logger.info(f"No valid candidates generated, stopping.")
                break
            if len(cands) == 1 and cands[0].is_eos and not cands[0].text:
                logger.info("EOS only, stopping.")
                break

            evaluated = self.evaluate_candidates(cands, query, steps)
            if not evaluated:
                logger.info(f"No valid candidates after evaluation, stopping.")
                break

            best_cand, best_immediate, new_step_values = evaluated[0]
            logger.info("   Candidate immediate values:")
            for i, (c, v, _) in enumerate(evaluated[: min(3, len(evaluated))], 1):
                logger.info(f"   Candidate {i}: {c.text[:50]}... (Value: {v:.4f})")

            # commit
            steps.append(best_cand.text)
            all_step_values = new_step_values
            cumulative_value += best_immediate
            logger.info(f"   Selected: {best_cand.text[:60]}...")
            logger.info(f"   Immediate Value: {best_immediate:.4f}")

            # carry chosen KV forward
            state = self._promote(best_cand)

            # free unchosen caches
            for c in cands:
                if c is not best_cand:
                    del c.past_kv, c.logits
            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\nFinal solution (Total Value: {cumulative_value:.8f}):")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")
            if i <= len(all_step_values):
                logger.info(f"           (Value: {all_step_values[i-1]:.4f})")

        return solution, all_step_values, cumulative_value

    def _initial_forward(self, query: str) -> RunningState:
        """
        Build the chat prompt once and seed the running KV cache + next-token logits.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        prompt = self.base_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.base_tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.base_device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.base_device)

        with torch.no_grad():
            if self.use_base_model_only:
                out = self.base_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                out = self.dpo_model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    head_index=None,  # ensemble
                )
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]

        # cleanup enc tensors we do not need anymore
        del enc, input_ids
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()

        return RunningState(past_kv=past_kv, logits=logits)

    def _sample_one_from_state(
        self,
        state: RunningState,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        max_new: int = 512,
        min_chars_before_boundary: int = 1,
        attempt_seed: Optional[int] = None,
    ) -> Candidate:
        """
        Sample a single contiguous chunk from the given state using 1-token steps.
        Stop at EOS or the first blank-line boundary.
        """
        # deep-copy the cache so we do not mutate the shared running cache
        local_past = clone_past_kv(state.past_kv)
        local_logits = state.logits.detach().clone()
        device = local_logits.device

        rng = torch.Generator(device=device)
        if attempt_seed is None:
            # reuse current seeding policy if not provided
            attempt_seed = int((time.time_ns() % 2_147_483_647))
        rng.manual_seed(attempt_seed)

        picked: List[int] = []
        emitted_chars = ""
        tokens_emitted = 0
        eos_id = self.base_tokenizer.eos_token_id

        while tokens_emitted < max_new:
            nid = _sample_next_token(
                local_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=rng,
            )
            picked.append(nid)
            tokens_emitted += 1

            # EOS check
            if eos_id is not None and nid == eos_id:
                text = self.base_tokenizer.decode(
                    picked, skip_special_tokens=True
                ).strip()
                return Candidate(
                    text=text,
                    token_ids=picked,
                    is_eos=True,
                    finish_reason="eos_token",
                    past_kv=local_past,
                    logits=local_logits,
                )

            # one-token forward step
            with torch.no_grad():
                if self.use_base_model_only:
                    out_step = self.base_model(
                        input_ids=torch.tensor(
                            [[nid]], device=device, dtype=torch.long
                        ),
                        past_key_values=local_past,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    out_step = self.dpo_model(
                        input_ids=torch.tensor(
                            [[nid]], device=device, dtype=torch.long
                        ),
                        past_key_values=local_past,
                        use_cache=True,
                        return_dict=True,
                        head_index=None,
                    )
            local_past = out_step.past_key_values
            local_logits = out_step.logits[:, -1, :]

            # boundary check: first double newline after a minimal length
            emitted_chars = self.base_tokenizer.decode(picked, skip_special_tokens=True)
            if (
                len(emitted_chars) >= min_chars_before_boundary
                and "\n\n" in emitted_chars
            ):
                text = emitted_chars.split("\n\n", 1)[0].strip()
                return Candidate(
                    text=text,
                    token_ids=picked,
                    is_eos=False,
                    finish_reason="boundary",
                    past_kv=local_past,
                    logits=local_logits,
                )

        # max_new limit
        text = self.base_tokenizer.decode(picked, skip_special_tokens=True).strip()
        return Candidate(
            text=text,
            token_ids=picked,
            is_eos=False,
            finish_reason="max_new_tokens",
            past_kv=local_past,
            logits=local_logits,
        )

    def _generate_candidates_from_state(
        self,
        state: RunningState,
        num_candidates: Optional[int] = None,
    ) -> List[Candidate]:
        """
        Propose multiple candidates from the same running state.
        Each candidate uses an independent RNG stream.
        """
        target = num_candidates if num_candidates is not None else self.num_candidates
        candidates: List[Candidate] = []
        attempts = 0
        max_attempts = max(target, 1) * 3

        # base seed consistent with your earlier logic
        base_seed = int((time.time_ns() % 2_147_483_647))

        while len(candidates) < target and attempts < max_attempts:
            attempts += 1
            attempt_seed = base_seed + attempts * 10007
            cand = self._sample_one_from_state(
                state,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                max_new=512,
                min_chars_before_boundary=1,
                attempt_seed=attempt_seed,
            )
            # keep non-empty, keep eos-only if nothing else is collected
            if cand.text:
                candidates.append(cand)
            elif cand.is_eos and not candidates:
                candidates.append(cand)

        return candidates

    def _promote(self, cand: Candidate) -> RunningState:
        """
        Adopt the chosen candidate's KV cache and logits as the new running state.
        """
        return RunningState(past_kv=cand.past_kv, logits=cand.logits)

    def _is_valid_step(self, step: str) -> bool:
        """Check if a step is valid."""
        step = step.strip()
        if len(step) == 0:
            return False
        if step.startswith("Step 0"):
            return False
        if step.count("assistant") > 1:
            return False
        return True


def set_random_seeds(base_seed=None, use_deterministic_seed=False):
    """Set all random seeds for reproducibility or true randomness."""
    import time
    import os

    if base_seed is None:
        # Use nanosecond precision + random component for better entropy
        import random as py_random

        # Initialize Python's random with OS entropy first
        py_random.seed()

        # Combine multiple sources for better randomness
        base_seed = (
            int(time.time_ns() % 1000000)  # Nanosecond precision
            + py_random.randint(0, 1000000)  # Random component
            + (os.getpid() * 7919)  # PID with prime multiplier
        ) % 2147483647  # Keep within int32 range

    # Handle seed generation based on user preference
    if use_deterministic_seed:
        # Use the original seed for deterministic behavior
        final_seed = base_seed
        logger.info(f"🎲 Using deterministic seed: {final_seed}")
    else:
        # Generate unique seed for each run to avoid identical outputs
        # Use current timestamp and process id to ensure uniqueness
        final_seed = base_seed + int(time.time() * 1000) % 100000 + os.getpid() % 1000
        logger.info(f"🎲 Using unique seed: {final_seed} (base seed: {base_seed})")

    # Set all random seeds
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)

    return final_seed


def run_dpo_ensemble_value_guided_inference(
    output_file: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    value_model_name: str = "Jennny/qwen-math-value-model-join",
    reward_model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
    dpo_checkpoint_path: str = None,  # Make optional
    num_heads: int = 2,
    num_samples: Optional[int] = None,
    split_index: Optional[int] = None,
    total_splits: int = 5,
    base_device_id: int = 0,
    value_device_id: int = 1,
    dtype: str = "bfloat16",
    cache_dir: Optional[str] = None,
    seed: int = 42,
    use_deterministic_seed: bool = False,
    base_only: bool = False,
    no_value_guidance: bool = False,
    use_prm_guidance: bool = False,
):
    """Run value-guided decoding with DPO ensemble model on MATH-500 dataset."""

    # Use seed parameter instead of None for reproducibility
    actual_seed = set_random_seeds(seed, use_deterministic_seed)

    # Load dataset
    logger.info("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    test_dataset = dataset["test"]

    all_samples = list(test_dataset)
    random.shuffle(all_samples)

    # Handle split processing
    if split_index is not None:
        if split_index < 0 or split_index >= total_splits:
            raise ValueError(f"split_index must be between 0 and {total_splits-1}")

        split_size = len(all_samples) // total_splits
        remainder = len(all_samples) % total_splits

        start_idx = split_index * split_size
        if split_index < remainder:
            start_idx += split_index
            end_idx = start_idx + split_size + 1
        else:
            start_idx += remainder
            end_idx = start_idx + split_size

        samples = all_samples[start_idx:end_idx]
        logger.info(
            f"Processing split {split_index}/{total_splits-1} with {len(samples)} samples (indices {start_idx}-{end_idx-1})"
        )
    else:
        samples = all_samples
        logger.info(f"Processing all {len(samples)} samples")

    # Apply additional sampling if requested
    if num_samples is not None and num_samples > 0:
        if num_samples > len(samples):
            logger.warning(
                f"Requested {num_samples} samples but only {len(samples)} available in this split"
            )
            num_samples = len(samples)
        samples = samples[:num_samples]
        logger.info(f"Using first {num_samples} samples from the split")

    # *** KEY CHANGE: Conditional checkpoint requirement ***
    if not base_only and dpo_checkpoint_path is None:
        raise ValueError("dpo_checkpoint_path is required when base_only=False")

    # Initialize decoder
    decoder = DPOEnsembleValueGuidedDecoder(
        base_model_name=base_model_name,
        dpo_checkpoint_path=dpo_checkpoint_path,  # Can be None for base_only
        value_model_name=value_model_name,
        reward_model_name=reward_model_name,
        num_heads=num_heads,
        base_device_id=base_device_id,
        value_device_id=value_device_id,
        dtype=dtype,
        cache_dir=cache_dir,
        use_value_guidance=not no_value_guidance and not use_prm_guidance,
        use_prm_guidance=use_prm_guidance,
        use_base_model_only=base_only,
    )

    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clear output file
    with open(output_file, "w") as f:
        pass

    start_time = time.time()

    # Process samples
    for idx, example in enumerate(tqdm(samples, desc="Processing samples")):
        problem = example["problem"]
        answer = example["answer"]
        subject = example.get("subject", "unknown")
        unique_id = example.get("unique_id", f"sample_{idx}")

        logger.info(f"\nProcessing sample {idx+1}/{len(samples)} (ID: {unique_id})")

        try:
            if base_only:
                if decoder.use_value_guidance:
                    # Base model with value guidance using hidden state sampling
                    model_response, step_values, cumulative_value = (
                        decoder.decode_base_hidden_with_value(problem)
                    )
                    model_type = "base_model_hidden_value_guided"
                elif decoder.use_prm_guidance:
                    # Base model with PRM guidance using hidden state sampling
                    model_response, step_values, cumulative_value = (
                        decoder.decode_base_hidden_with_prm(problem)
                    )
                    model_type = "base_model_hidden_prm_guided"
                else:
                    # Base model with hidden state sampling, no value guidance
                    model_response, step_values, cumulative_value = (
                        decoder.decode_base_hidden_no_value(problem)
                    )
                    model_type = "base_model_hidden_no_value"
            else:
                if decoder.use_value_guidance:
                    # DPO ensemble with value guidance
                    model_response, step_values, cumulative_value = (
                        decoder.decode_hidden(problem)
                    )
                    model_type = "dpo_ensemble_value_guided"
                elif decoder.use_prm_guidance:
                    # DPO ensemble with PRM guidance (reuse hidden generator + PRM scorer)
                    model_response, step_values, cumulative_value = (
                        decoder.decode_base_hidden_with_prm(problem)
                    )
                    model_type = "dpo_ensemble_prm_guided"
                else:
                    # DPO ensemble without value guidance
                    model_response, step_values, cumulative_value = (
                        decoder.decode_base_hidden(problem)
                    )
                    model_type = "dpo_ensemble_no_value"

            # Create output record
            output_record = {
                "id": f"test/{subject.lower()}/{unique_id}.json",
                "question": problem,
                "ground_truth": answer,
                "model_response": model_response,
                "step_values": step_values,
                "cumulative_value": cumulative_value,
                "model_type": model_type,
                "num_heads": num_heads,
                "split_info": (
                    {
                        "split_index": split_index,
                        "total_splits": total_splits,
                    }
                    if split_index is not None
                    else None
                ),
            }

            # Write to file
            with open(output_file, "a") as f:
                f.write(json.dumps(output_record) + "\n")

            logger.info(f"Completed sample {idx+1}/{len(samples)}")

        except Exception as e:
            logger.error(f"Error processing sample {idx+1}: {e}")
            # Write error record
            error_record = {
                "id": f"test/{subject.lower()}/{unique_id}.json",
                "question": problem,
                "ground_truth": answer,
                "model_response": "",
                "error": str(e),
                "model_type": (
                    "base_model_hidden_value_guided"
                    if base_only and decoder.use_value_guidance
                    else (
                        "base_model_hidden_prm_guided"
                        if base_only and decoder.use_prm_guidance
                        else (
                            "base_model_hidden_no_value"
                            if base_only
                            else (
                                "dpo_ensemble_value_guided"
                                if decoder.use_value_guidance
                                else (
                                    "dpo_ensemble_prm_guided"
                                    if decoder.use_prm_guidance
                                    else "dpo_ensemble_no_value"
                                )
                            )
                        )
                    )
                ),
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(error_record) + "\n")

        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

        # Log progress
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(samples) - idx - 1)

            split_info = (
                f" (Split {split_index}/{total_splits-1})"
                if split_index is not None
                else ""
            )
            logger.info(f"Processed {idx + 1}/{len(samples)} examples{split_info}")
            logger.info(
                f"Avg time per example: {avg_time:.2f}s, Est. time remaining: {remaining:.2f}s"
            )

    total_time = time.time() - start_time
    split_info = (
        f" for split {split_index}/{total_splits-1}" if split_index is not None else ""
    )
    logger.info(
        f"Completed inference on {len(samples)} examples{split_info} in {total_time:.2f}s"
    )
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Value-guided decoding with ensemble multi-head DPO model"
    )

    # Model configuration
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Use base model instead of DPO ensemble",
    )
    parser.add_argument(
        "--no_value_guidance",
        action="store_true",
        help="Disable value model guidance (works with both base model and DPO ensemble)",
    )
    parser.add_argument(
        "--use_prm_guidance",
        action="store_true",
        help="Enable PRM (reward) model guidance instead of value model guidance",
    )
    parser.add_argument(
        "--dpo_checkpoint_path",
        type=str,
        # Remove required=True
        help="Path to DPO checkpoint (S3, HF Hub, or local). Not needed when --base_only is used.",
    )
    parser.add_argument(
        "--value_model_name",
        type=str,
        default="Jennny/qwen-math-value-model-join",
        help="Value model name",
    )
    parser.add_argument(
        "--reward_model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="PRM (reward) model name",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of heads in the DPO model",
    )

    # GPU configuration
    parser.add_argument(
        "--base_device_id",
        type=int,
        default=0,
        help="GPU ID for DPO ensemble model",
    )
    parser.add_argument(
        "--value_device_id",
        type=int,
        default=1,
        help="GPU ID for value model",
    )

    # Data configuration
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--split_index",
        type=int,
        help="Which split to process (0 to total_splits-1)",
    )
    parser.add_argument(
        "--total_splits",
        type=int,
        default=5,
        help="Total number of splits",
    )

    # Other parameters
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path (JSONL format)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Model precision",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to cache downloaded files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_deterministic_seed",
        action="store_true",
        help="Use the same seed for all runs (deterministic behavior)",
    )

    args = parser.parse_args()

    # *** KEY CHANGE: Conditional checkpoint validation ***
    if not args.base_only and args.dpo_checkpoint_path is None:
        parser.error("--dpo_checkpoint_path is required when --base_only is not used")

    # Run inference
    run_dpo_ensemble_value_guided_inference(
        output_file=args.output_file,
        base_model_name=args.base_model_name,
        value_model_name=args.value_model_name,
        reward_model_name=args.reward_model_name,
        dpo_checkpoint_path=args.dpo_checkpoint_path,  # Can be None
        num_heads=args.num_heads,
        num_samples=args.num_samples,
        split_index=args.split_index,
        total_splits=args.total_splits,
        base_device_id=args.base_device_id,
        value_device_id=args.value_device_id,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        seed=args.seed,
        use_deterministic_seed=args.use_deterministic_seed,
        base_only=args.base_only,
        no_value_guidance=args.no_value_guidance,
        use_prm_guidance=args.use_prm_guidance,
    )


if __name__ == "__main__":
    main()
