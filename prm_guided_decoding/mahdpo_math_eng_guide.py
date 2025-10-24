import os

# Remove any leftover DDP env-vars
for v in ("LOCAL_RANK", "WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(v, None)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
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

try:
    from multihead_model import MultiHeadCausalLM

    MDPO_AVAILABLE = True
except ImportError:
    MDPO_AVAILABLE = False
    print("Warning: multihead_model not found. MDPO functionality will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


class UnifiedRewardModel:
    """
    Wrapper for unified reward model that provides step-level scoring
    Compatible with the existing value guidance interface

    This version is designed for binary classification reward models (2 labels).
    Returns the probability of the positive class (label 1) as the reward score.
    """

    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.step_sep_token_id = None

    def to(self, device):
        """Move model to device"""
        self.base_model = self.base_model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode"""
        self.base_model.eval()
        return self

    def parameters(self):
        """Return model parameters"""
        return self.base_model.parameters()

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """Load unified reward model from pretrained"""
        # Extract tokenizer and base_model_name if provided in kwargs
        tokenizer = kwargs.pop("tokenizer", None)
        base_model_name = kwargs.pop("base_model_name", None)

        # Load the sequence classification model (binary classification)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            problem_type="single_label_classification",
            **kwargs,
        )

        # Load tokenizer if not provided
        if tokenizer is None:
            # For unified reward models, prioritize the base model used during training
            # Since the saved unified model doesn't have a tokenizer, load from meta-llama/Llama-3.1-8B
            tokenizer_sources = []

            # First try the base model used for training unified models
            tokenizer_sources.append("meta-llama/Llama-3.1-8B")

            # Then try the provided base_model_name if different
            if base_model_name and base_model_name != "meta-llama/Llama-3.1-8B":
                tokenizer_sources.append(base_model_name)

            # Try the model path itself (unlikely to work for unified models but worth trying)
            tokenizer_sources.append(model_name_or_path)

            tokenizer = None
            for source in tokenizer_sources:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        source, trust_remote_code=True
                    )
                    logger.info(f"Successfully loaded tokenizer from: {source}")
                    break
                except Exception as e:
                    logger.warning(
                        f"WARNING: Could not load tokenizer from {source}: {e}"
                    )
                    continue

            if tokenizer is None:
                raise ValueError("Could not load tokenizer from any source")

        # Set up tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Add default chat template if missing
        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n\n{% endif %}{% endfor %}"
            logger.info("🔧 Added default chat template to tokenizer")

        model = cls(base_model, tokenizer)

        # Set step separator token ID if available
        step_sep_token = "<extra_0>"
        try:
            model.step_sep_token_id = tokenizer.encode(step_sep_token)[0]
            logger.info(f"Set step separator token ID: {model.step_sep_token_id}")
        except:
            logger.warning(
                f"WARNING: Could not set step separator token ID for {step_sep_token}"
            )
            model.step_sep_token_id = None

        return model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through the unified model (binary classification)"""
        # Get model device
        model_device = next(self.base_model.parameters()).device

        # Move inputs to model device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if attention_mask is not None and attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)

        # Forward pass
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        # For binary classification, extract probability of positive class (class 1)
        logits = outputs.logits  # Shape: [batch_size, 2]
        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
        scores = probs[
            :, 1:2
        ]  # Extract probability of class 1, keep dimension [batch_size, 1]

        return type(
            "UnifiedOutput",
            (),
            {
                "logits": scores,  # Return positive class probability as logits for compatibility
                "last_hidden_state": getattr(outputs, "hidden_states", None),
            },
        )()


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
    logits: torch.Tensor  # [1, vocab]


@dataclass
class Candidate:
    text: str
    token_ids: List[int]
    is_eos: bool
    finish_reason: str
    past_kv: object
    logits: torch.Tensor


class UnifiedStepByStepDecoder:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dpo_checkpoint_path: str = None,
        unified_reward_model_name: str = "Jennny/unified_rm_1e5_1600",
        num_heads: int = 2,
        max_steps: int = 20,
        num_candidates: int = 5,
        base_device_id: int = 0,
        reward_device_id: int = 1,
        dtype: str = "bfloat16",
        cache_dir: Optional[str] = None,
        use_reward_guidance: bool = False,
        use_mdpo_model: bool = False,
        use_hidden_states: bool = True,
        candidate_selection: str = "random",  # "random", "best", "unified_guided"
        mdpo_head_mode: str = "ensemble",  # "base", "head_0", "head_1", "ensemble"
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        seed: int = 42,
    ):
        self.max_steps = max_steps
        self.num_candidates = num_candidates
        self.num_heads = num_heads
        self.use_reward_guidance = use_reward_guidance
        self.use_mdpo_model = use_mdpo_model and MDPO_AVAILABLE
        self.use_hidden_states = use_hidden_states
        self.candidate_selection = candidate_selection
        self.mdpo_head_mode = mdpo_head_mode
        self.seed = seed

        # If no reward guidance, use only 1 candidate to match value guide behavior
        if not self.use_reward_guidance and self.candidate_selection == "random":
            self.num_candidates = 1
            logger.info(
                "🔧 No reward guidance detected: using 1 candidate to match base model behavior"
            )

        # Set devices
        self.base_device = torch.device(f"cuda:{base_device_id}")
        self.reward_device = torch.device(f"cuda:{reward_device_id}")

        # Set torch dtype
        self.torch_dtype = getattr(torch, dtype)

        # Configuration logging
        if self.use_mdpo_model:
            logger.info(f"MDPO Model + Hidden State Setup:")
            logger.info(
                f"   MDPO model ({self.mdpo_head_mode}) -> GPU {base_device_id}"
            )
        else:
            if self.use_hidden_states:
                logger.info(f"Base Model + Hidden State Setup:")
            else:
                logger.info(f"Base Model + vLLM Setup:")
            logger.info(f"   Base model -> GPU {base_device_id}")

        if self.use_reward_guidance:
            logger.info(f"   Unified reward model -> GPU {reward_device_id}")

        # Load tokenizer
        logger.info(f"Loading tokenizer: {base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if self.base_tokenizer.pad_token_id is None:
            self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

        # Load base model or MDPO model
        if self.use_mdpo_model:
            logger.info(f"Loading base model for MDPO: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )

            logger.info(f"Creating multi-head DPO model with {num_heads} heads")
            self.mdpo_model = MultiHeadCausalLM(base_model, num_heads=num_heads).to(
                self.base_device
            )
            self.base_model = None

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
                    self.mdpo_model.load_state_dict(state_dict, strict=False)
                else:
                    self.mdpo_model.load_state_dict(checkpoint, strict=False)

                logger.info("Successfully loaded DPO checkpoint")

            self.mdpo_model.eval()
        else:
            logger.info(f"Loading base model: {base_model_name}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map={"": self.base_device},
            ).eval()
            self.mdpo_model = None

        # Initialize vLLM if not using hidden states
        if not self.use_hidden_states:
            logger.info(f"Initializing vLLM with {base_model_name}")
            self.vllm_model = LLM(
                model=base_model_name,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
        else:
            self.vllm_model = None

        # Configure sampling parameters for vLLM (when used)
        if self.vllm_model:
            self.generation_params = VLLMSamplingParams(
                temperature=1.0,
                max_tokens=512,
                top_p=1.0,
                top_k=50,
                stop=["\n\n"],
                include_stop_str_in_output=True,
                n=self.num_candidates,
            )

        # Load unified reward model if guidance is enabled
        if self.use_reward_guidance:
            logger.info(f"Loading unified reward model on GPU {reward_device_id}...")
            try:
                self.unified_reward_model = UnifiedRewardModel.from_pretrained(
                    unified_reward_model_name,
                    base_model_name=base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map={"": self.reward_device},
                    trust_remote_code=True,
                ).eval()
                self.reward_tokenizer = self.unified_reward_model.tokenizer
                logger.info(
                    f"Successfully loaded unified reward model: {unified_reward_model_name}"
                )
            except Exception as e:
                logger.error(f"ERROR: Failed to load unified reward model: {e}")
                raise
        else:
            self.unified_reward_model = None
            self.reward_tokenizer = None

        self.system_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        logger.info(f"Setup complete!")

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

    def generate_candidates_vllm(
        self,
        query: str,
        previous_steps: List[str],
        num_candidates: Optional[int] = None,
    ) -> List[str]:
        """Generate candidate steps using vLLM."""
        target_candidates = (
            num_candidates if num_candidates is not None else self.num_candidates
        )

        logger.info(f"🔧 Using vLLM to generate {target_candidates} candidate(s)")

        formatted_prompt = self.build_conversation_prompt(query, previous_steps)

        generation_params = VLLMSamplingParams(
            temperature=1.0,
            max_tokens=512,
            top_p=1.0,
            top_k=50,
            stop=["\n\n"],
            include_stop_str_in_output=True,
            n=target_candidates,
        )

        outputs = self.vllm_model.generate([formatted_prompt], generation_params)

        candidates = []
        for output in outputs[0].outputs:
            text = output.text.strip()
            if self._is_valid_step(text):
                candidates.append(text)

        logger.info(f"Generated {len(candidates)} valid candidates using vLLM")
        return candidates

    def generate_candidates_hidden_state(
        self,
        query: str,
        previous_steps: List[str],
        num_candidates: Optional[int] = None,
    ) -> List[str]:
        """Generate candidates using hidden state caching method."""
        target_candidates = (
            num_candidates if num_candidates is not None else self.num_candidates
        )

        logger.info(
            f"🔧 Using hidden state caching to generate {target_candidates} candidate(s)"
        )

        # Build prompt once
        prompt = self.build_conversation_prompt(query, previous_steps)
        enc = self.base_tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.base_device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.base_device)

        # Choose model based on configuration
        if self.use_mdpo_model:
            model = self.mdpo_model
        else:
            model = self.base_model

        # Initial forward to get cache and next token logits
        with torch.no_grad():
            if self.use_mdpo_model:
                # Handle different MDPO head modes
                if self.mdpo_head_mode == "base":
                    head_index = None  # Use base model without DPO heads
                elif self.mdpo_head_mode == "head_0":
                    head_index = 0
                elif self.mdpo_head_mode == "head_1":
                    head_index = 1
                elif self.mdpo_head_mode == "ensemble":
                    head_index = None  # Use ensemble
                else:
                    head_index = None  # Default to ensemble

                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    head_index=head_index,
                )
            else:
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

        main_past = out.past_key_values
        main_logits = out.logits[:, -1, :]

        # Sampling settings
        temperature = 1.0
        top_p = 1.0
        top_k = 50

        candidates = []
        attempts = 0
        max_attempts = target_candidates * 3

        # Generate candidates using hidden state caching
        while len(candidates) < target_candidates and attempts < max_attempts:
            attempts += 1

            # Clone the cached state for this candidate
            local_past = clone_past_kv(main_past)
            local_logits = main_logits.detach().clone()
            picked = []
            device = local_logits.device
            gen = torch.Generator(device=device)
            # gen.manual_seed(42 + attempts)
            gen.manual_seed(self.seed + len(candidates) * 10007 + attempts)

            # Generate one candidate
            new_text = ""
            tokens_emitted = 0
            max_new = 512
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

                # Advance one step
                with torch.no_grad():
                    if self.use_mdpo_model:
                        # Handle MDPO head mode for incremental generation
                        if self.mdpo_head_mode == "base":
                            head_index = None
                        elif self.mdpo_head_mode == "head_0":
                            head_index = 0
                        elif self.mdpo_head_mode == "head_1":
                            head_index = 1
                        elif self.mdpo_head_mode == "ensemble":
                            head_index = None
                        else:
                            head_index = None

                        out_step = model(
                            input_ids=torch.tensor(
                                [[nid]], device=device, dtype=torch.long
                            ),
                            past_key_values=local_past,
                            use_cache=True,
                            return_dict=True,
                            head_index=head_index,
                        )
                    else:
                        out_step = model(
                            input_ids=torch.tensor(
                                [[nid]], device=device, dtype=torch.long
                            ),
                            past_key_values=local_past,
                            use_cache=True,
                            return_dict=True,
                        )
                local_past = out_step.past_key_values
                local_logits = out_step.logits[:, -1, :]

                # Decode and check for stopping condition
                new_text = self.base_tokenizer.decode(picked, skip_special_tokens=True)
                if "\n\n" in new_text:
                    new_text = new_text.split("\n\n", 1)[0].strip()
                    break

            if self._is_valid_step(new_text.strip()):
                candidates.append(new_text.strip())

        # Cleanup
        del out, main_past, main_logits, enc
        torch.cuda.empty_cache()

        logger.info(
            f"Generated {len(candidates)} valid candidates using hidden state caching"
        )
        return candidates

    def calculate_unified_reward(
        self, steps: List[str], query: str
    ) -> Tuple[float, List[float]]:
        """Calculate unified reward scores for the conversation."""
        if not steps or self.unified_reward_model is None:
            return 0.0, []

        # Build conversation exactly like training data format
        conversation = f"User: {query}\n\nAssistant: "

        # Add steps incrementally with proper spacing
        for i, step in enumerate(steps):
            if i == 0:
                conversation += step  # First step directly after "Assistant: "
            else:
                conversation += f"\n\n{step}"  # Subsequent steps with \n\n spacing

        input_ids = self.reward_tokenizer.encode(conversation, return_tensors="pt").to(
            self.reward_device
        )
        attention_mask = torch.ones_like(input_ids).to(self.reward_device)

        with torch.no_grad():
            # Get unified model prediction for full conversation
            outputs = self.unified_reward_model.forward(input_ids, attention_mask)
            conversation_score = outputs.logits.squeeze().cpu().item()

        # Clean up tensors
        del input_ids, attention_mask
        torch.cuda.empty_cache()

        # Return the current conversation score for each step (for logging purposes)
        # Each step gets the current conversation score, not split evenly
        step_scores = [conversation_score] * len(steps) if steps else []

        return conversation_score, step_scores

    def evaluate_candidates_unified(
        self, candidates: List[str], query: str, previous_steps: List[str]
    ) -> List[Tuple[str, float, List[float]]]:
        """Evaluate candidates using unified reward model."""
        results = []

        for candidate in candidates:
            if not self._is_valid_step(candidate):
                continue

            test_steps = previous_steps + [candidate]
            new_score, new_step_scores = self.calculate_unified_reward(
                test_steps, query
            )

            # Use absolute score instead of improvement
            results.append((candidate, new_score, new_step_scores))

        # Sort by absolute score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # Unified-RM evaluator for Candidate objects
    def evaluate_candidates_unified_candidates(
        self, candidates: List["Candidate"], query: str, previous_steps: List[str]
    ) -> List[Tuple["Candidate", float, List[float]]]:
        results: List[Tuple[Candidate, float, List[float]]] = []
        for cand in candidates:
            if not self._is_valid_step(cand.text):
                continue
            test_steps = previous_steps + [cand.text]
            abs_score, step_scores = self.calculate_unified_reward(test_steps, query)
            results.append((cand, abs_score, step_scores))
            del test_steps
            torch.cuda.empty_cache()
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def decode(self, query: str) -> Tuple[str, List[float], float]:
        """Main decoding method."""
        steps: List[str] = []
        all_step_scores: List[float] = []
        cumulative_score: float = 0.0

        method_desc = f"Unified Step-by-Step Decoder"
        if self.use_mdpo_model:
            method_desc += f" (MDPO-{self.mdpo_head_mode})"
        elif self.use_hidden_states:
            method_desc += " (Hidden State)"
        else:
            method_desc += " (vLLM)"

        if self.use_reward_guidance:
            method_desc += f" + Unified Reward Guidance ({self.candidate_selection})"
        else:
            method_desc += f" ({self.candidate_selection} selection)"

        logger.info(f"\nStarting {method_desc} for query:\n{query}\n")

        # Hidden-state path: seed once and keep a running KV cache across steps
        if self.use_hidden_states:
            state = self._initial_forward_state(query)

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
                        f"Current solution (Cumulative Score: {cumulative_score:.4f}):"
                    )
                    for i, s in enumerate(steps, 1):
                        logger.info(f"   Step {i}: {s[:60]}...")
                        if i <= len(all_step_scores):
                            logger.info(
                                f"           (Score: {all_step_scores[i-1]:.4f})"
                            )

                cands: List[Candidate] = self._generate_candidates_from_state(
                    state, self.num_candidates
                )
                if not cands:
                    logger.info(f"WARNING: No valid candidates generated, stopping.")
                    break
                if len(cands) == 1 and cands[0].is_eos and not cands[0].text:
                    logger.info("WARNING: EOS only, stopping.")
                    break

                # Selection
                if (
                    self.use_reward_guidance
                    and self.candidate_selection == "unified_guided"
                ):
                    evald = self.evaluate_candidates_unified_candidates(
                        cands, query, steps
                    )
                    if not evald:
                        logger.info(
                            f"WARNING: No valid candidates after evaluation, stopping."
                        )
                        break
                    best_cand, best_abs_score, step_scores = evald[0]
                    logger.info("   Candidate absolute scores:")
                    for i, (c, sc, _) in enumerate(evald[: min(3, len(evald))], 1):
                        logger.info(
                            f"   Candidate {i}: {c.text[:50]}... (Score: {sc:.4f})"
                        )
                    if step_scores:
                        all_step_scores.append(step_scores[-1])
                    cumulative_score = best_abs_score
                    logger.info(f"   Selected: {best_cand.text[:60]}...")
                    logger.info(f"   Absolute Score: {best_abs_score:.4f}")
                elif self.candidate_selection == "best":
                    # simple heuristic: take the first candidate
                    best_cand = cands[0]
                    logger.info(
                        f"   Selected (best heuristic): {best_cand.text[:60]}..."
                    )
                else:
                    import random as _r

                    best_cand = _r.choice(cands)
                    logger.info(f"   Selected (random): {best_cand.text[:60]}...")

                steps.append(best_cand.text)

                # Promote the chosen candidate cache
                state = self._promote(best_cand)

                # Free unchosen caches
                for c in cands:
                    if c is not best_cand:
                        del c.past_kv, c.logits
                torch.cuda.empty_cache()
                gc.collect()

            solution = "\n\n".join(steps)
            logger.info(f"\n💡 Final solution (Total Score: {cumulative_score:.4f}):")
            for i, step in enumerate(steps, 1):
                logger.info(f"   Step {i}: {step}")
                if i <= len(all_step_scores):
                    logger.info(f"           (Score: {all_step_scores[i-1]:.4f})")
            return solution, all_step_scores, cumulative_score

        # vLLM path: unchanged
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
                    f"Current solution (Cumulative Score: {cumulative_score:.4f}):"
                )
                for i, s in enumerate(steps, 1):
                    logger.info(f"   Step {i}: {s[:60]}...")
                    if i <= len(all_step_scores):
                        logger.info(f"           (Score: {all_step_scores[i-1]:.4f})")

            candidates = self.generate_candidates_vllm(query, steps)
            if not candidates:
                logger.info(f"WARNING: No valid candidates generated, stopping.")
                break

            if (
                self.use_reward_guidance
                and self.candidate_selection == "unified_guided"
            ):
                evaluated_candidates = self.evaluate_candidates_unified(
                    candidates, query, steps
                )
                if not evaluated_candidates:
                    logger.info(
                        f"WARNING: No valid candidates after evaluation, stopping."
                    )
                    break
                best_candidate, best_absolute_score, new_step_scores = (
                    evaluated_candidates[0]
                )
                logger.info("   Candidate absolute scores:")
                for i, (cand, abs_score, _) in enumerate(
                    evaluated_candidates[: min(3, len(evaluated_candidates))], 1
                ):
                    logger.info(
                        f"   Candidate {i}: {cand[:50]}... (Score: {abs_score:.4f})"
                    )
                if new_step_scores:
                    all_step_scores.append(new_step_scores[-1])
                cumulative_score = best_absolute_score
                logger.info(f"   Selected: {best_candidate[:60]}...")
                logger.info(f"   Absolute Score: {best_absolute_score:.4f}")
            elif self.candidate_selection == "best":
                best_candidate = candidates[0]
                logger.info(f"   Selected (best heuristic): {best_candidate[:60]}...")
            else:
                best_candidate = random.choice(candidates)
                logger.info(f"   Selected (random): {best_candidate[:60]}...")

            steps.append(best_candidate)

            torch.cuda.empty_cache()
            gc.collect()

        solution = "\n\n".join(steps)
        logger.info(f"\n💡 Final solution (Total Score: {cumulative_score:.4f}):")
        for i, step in enumerate(steps, 1):
            logger.info(f"   Step {i}: {step}")
            if i <= len(all_step_scores):
                logger.info(f"           (Score: {all_step_scores[i-1]:.4f})")
        return solution, all_step_scores, cumulative_score

    # Model/head picker used by initial and step forwards
    def _active_model_and_head(self):
        if self.use_mdpo_model:
            model = self.mdpo_model
            if self.mdpo_head_mode == "base":
                head_index = None
            elif self.mdpo_head_mode == "head_0":
                head_index = 0
            elif self.mdpo_head_mode == "head_1":
                head_index = 1
            else:
                head_index = None  # ensemble or default
            return model, head_index, self.base_device
        else:
            return self.base_model, None, self.base_device

    # Seed the running cache once from the chat prompt
    def _initial_forward_state(self, query: str) -> RunningState:
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

        model, head_index, _ = self._active_model_and_head()
        with torch.no_grad():
            if self.use_mdpo_model:
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    head_index=head_index,
                )
            else:
                out = model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]

        del enc, input_ids
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()

        return RunningState(past_kv=past_kv, logits=logits)

    # Sample a single candidate chunk from a given KV state
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
    ) -> "Candidate":
        local_past = clone_past_kv(state.past_kv)
        local_logits = state.logits.detach().clone()
        model, head_index, device = self._active_model_and_head()

        rng = torch.Generator(device=local_logits.device)
        if attempt_seed is None:
            attempt_seed = int((time.time_ns() % 2_147_483_647))
        rng.manual_seed(attempt_seed)

        picked: List[int] = []
        eos_id = self.base_tokenizer.eos_token_id
        emitted_text = ""
        tokens_emitted = 0

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

            with torch.no_grad():
                if self.use_mdpo_model:
                    out_step = model(
                        input_ids=torch.tensor(
                            [[nid]], device=device, dtype=torch.long
                        ),
                        past_key_values=local_past,
                        use_cache=True,
                        return_dict=True,
                        head_index=head_index,
                    )
                else:
                    out_step = model(
                        input_ids=torch.tensor(
                            [[nid]], device=device, dtype=torch.long
                        ),
                        past_key_values=local_past,
                        use_cache=True,
                        return_dict=True,
                    )
            local_past = out_step.past_key_values
            local_logits = out_step.logits[:, -1, :]

            emitted_text = self.base_tokenizer.decode(picked, skip_special_tokens=True)
            if (
                len(emitted_text) >= min_chars_before_boundary
                and "\n\n" in emitted_text
            ):
                text = emitted_text.split("\n\n", 1)[0].strip()
                return Candidate(
                    text=text,
                    token_ids=picked,
                    is_eos=False,
                    finish_reason="boundary",
                    past_kv=local_past,
                    logits=local_logits,
                )

        text = self.base_tokenizer.decode(picked, skip_special_tokens=True).strip()
        return Candidate(
            text=text,
            token_ids=picked,
            is_eos=False,
            finish_reason="max_new_tokens",
            past_kv=local_past,
            logits=local_logits,
        )

    # Propose N candidates from the same running cache
    def _generate_candidates_from_state(
        self, state: RunningState, n: Optional[int] = None
    ) -> List["Candidate"]:
        target = n if n is not None else self.num_candidates
        cands: List[Candidate] = []
        attempts = 0
        max_attempts = max(target, 1) * 3
        base_seed = self.seed if isinstance(self.seed, int) else 0
        base_seed = int((base_seed + time.time_ns()) % 2_147_483_647)

        while len(cands) < target and attempts < max_attempts:
            attempts += 1
            cand = self._sample_one_from_state(
                state,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                max_new=512,
                min_chars_before_boundary=1,
                attempt_seed=base_seed + attempts * 10007,
            )
            if cand.text:
                cands.append(cand)
            elif cand.is_eos and not cands:
                cands.append(cand)
        return cands

    # Adopt the chosen candidate's cache as the new running state
    def _promote(self, cand: "Candidate") -> RunningState:
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


def run_unified_inference(
    output_file: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dpo_checkpoint_path: str = None,
    unified_reward_model_name: str = "Jennny/unified_rm_1e5_1600",
    num_heads: int = 2,
    num_candidates: int = 5,
    num_samples: Optional[int] = None,
    split_index: Optional[int] = None,
    total_splits: int = 5,
    base_device_id: int = 0,
    reward_device_id: int = 1,
    dtype: str = "bfloat16",
    cache_dir: Optional[str] = None,
    seed: int = 42,
    use_deterministic_seed: bool = False,
    use_reward_guidance: bool = False,
    use_mdpo_model: bool = False,
    use_hidden_states: bool = True,
    candidate_selection: str = "random",
    mdpo_head_mode: str = "ensemble",
):
    """Run unified reward guided decoding."""
    # Handle seed generation based on user preference
    if use_deterministic_seed:
        # Use the original seed for deterministic behavior
        final_seed = seed
        logger.info(f"Using deterministic seed: {final_seed}")
    else:
        # Generate unique seed for each run to avoid identical outputs
        # Use current timestamp and process id to ensure uniqueness
        import time
        import os

        # Create a unique seed based on timestamp, process id, and original seed
        # This ensures different runs produce different outputs while maintaining
        # some reproducibility within the same run
        final_seed = seed + int(time.time() * 1000) % 100000 + os.getpid() % 1000
        logger.info(f"Using unique seed: {final_seed} (base seed: {seed})")

    # Set random seeds
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)

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
            f"Processing split {split_index}/{total_splits-1} with {len(samples)} samples"
        )
    else:
        samples = all_samples
        logger.info(f"Processing all {len(samples)} samples")

    # Apply additional sampling if requested
    if num_samples is not None and num_samples > 0:
        if num_samples > len(samples):
            logger.warning(
                f"Requested {num_samples} samples but only {len(samples)} available"
            )
            num_samples = len(samples)
        samples = samples[:num_samples]
        logger.info(f"Using first {num_samples} samples from the split")

    # Initialize decoder
    decoder = UnifiedStepByStepDecoder(
        base_model_name=base_model_name,
        dpo_checkpoint_path=dpo_checkpoint_path,
        unified_reward_model_name=unified_reward_model_name,
        num_heads=num_heads,
        num_candidates=num_candidates,
        base_device_id=base_device_id,
        reward_device_id=reward_device_id,
        dtype=dtype,
        cache_dir=cache_dir,
        use_reward_guidance=use_reward_guidance,
        use_mdpo_model=use_mdpo_model,
        use_hidden_states=use_hidden_states,
        candidate_selection=candidate_selection,
        mdpo_head_mode=mdpo_head_mode,
        seed=final_seed,
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
            # Run decoding
            model_response, step_scores, cumulative_score = decoder.decode(problem)

            # Determine model type string
            model_type = "unified_step_by_step"
            if decoder.use_mdpo_model:
                model_type += f"_mdpo_{decoder.mdpo_head_mode}"
            elif decoder.use_hidden_states:
                model_type += "_hidden"
            else:
                model_type += "_vllm"

            if decoder.use_reward_guidance:
                model_type += f"_unified_reward_{decoder.candidate_selection}"
            else:
                model_type += f"_{decoder.candidate_selection}"

            # Create output record
            output_record = {
                "id": f"test/{subject.lower()}/{unique_id}.json",
                "question": problem,
                "ground_truth": answer,
                "model_response": model_response,
                "step_scores": step_scores,
                "cumulative_score": cumulative_score,
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
                "model_type": "error",
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
        description="Unified reward guided step-by-step decoding"
    )

    # Model configuration
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--dpo_checkpoint_path",
        type=str,
        help="Path to DPO checkpoint",
    )
    parser.add_argument(
        "--unified_reward_model_name",
        type=str,
        default="Jennny/unified_rm_1e5_1600",
        help="Unified reward model name",
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
        help="GPU ID for base/DPO model",
    )
    parser.add_argument(
        "--reward_device_id",
        type=int,
        default=1,
        help="GPU ID for unified reward model",
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

    # Method configuration
    parser.add_argument(
        "--use_reward_guidance",
        action="store_true",
        help="Use unified reward model guidance",
    )
    parser.add_argument(
        "--use_mdpo_model",
        action="store_true",
        help="Use MDPO model instead of base model",
    )
    parser.add_argument(
        "--use_hidden_states",
        action="store_true",
        default=True,
        help="Use hidden state caching (default: True)",
    )
    parser.add_argument(
        "--candidate_selection",
        type=str,
        default="random",
        choices=["random", "best", "unified_guided"],
        help="Candidate selection method",
    )
    parser.add_argument(
        "--mdpo_head_mode",
        type=str,
        default="ensemble",
        choices=["base", "head_0", "head_1", "ensemble"],
        help="MDPO head mode: base (no DPO), head_0, head_1, or ensemble",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=5,
        help="Number of candidates to generate per step",
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

    # Determine use_hidden_states based on flags
    use_hidden_states = (
        not hasattr(args, "no_hidden_states") or not args.no_hidden_states
    )
    if hasattr(args, "use_vllm") and args.use_vllm:
        use_hidden_states = False

    # Run inference
    run_unified_inference(
        output_file=args.output_file,
        base_model_name=args.base_model_name,
        dpo_checkpoint_path=args.dpo_checkpoint_path,
        unified_reward_model_name=args.unified_reward_model_name,
        num_heads=args.num_heads,
        num_candidates=args.num_candidates,
        num_samples=args.num_samples,
        split_index=args.split_index,
        total_splits=args.total_splits,
        base_device_id=args.base_device_id,
        reward_device_id=args.reward_device_id,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        seed=args.seed,
        use_deterministic_seed=args.use_deterministic_seed,
        use_reward_guidance=args.use_reward_guidance,
        use_mdpo_model=args.use_mdpo_model,
        use_hidden_states=use_hidden_states,
        candidate_selection=args.candidate_selection,
        mdpo_head_mode=args.mdpo_head_mode,
    )


if __name__ == "__main__":
    main()
