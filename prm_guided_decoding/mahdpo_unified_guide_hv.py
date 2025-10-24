import os
import multiprocessing

# Set multiprocessing start method to 'spawn' before any CUDA operations
# This prevents CUDA re-initialization errors with vLLM
multiprocessing.set_start_method("spawn", force=True)

# Set environment variables to help with CUDA/multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import List, Tuple, Optional, Dict
import json
import random
import gc
import re
from datasets import load_dataset
import argparse

# Add Hugging Face Hub support
from huggingface_hub import hf_hub_download
import copy
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# UF-aware sentence boundary stopping criteria
_LIST_MARKER_LINE_RE = re.compile(
    r"""(?mx) ^\s*                # start of line
         (?: [-*•]                # bullet
           | \d+[\.)]             # 1.  2)
           | [A-Za-z][\.)]        # A.  b)
           | (?:Step|Stage)\s*\d+[:.]  # Step 1:  Stage 2.
         )
         \s* $                    # nothing else on the line
    """
)

_SENT_END_RE = re.compile(r"""[.!?](?:["'\)\]]+)?\s*$""")

_ABBREV = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "cf.",
    "et al.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
}


def _normalize_join(prev: str, cur: str) -> str:
    s = prev.strip() + cur.rstrip()
    return s


class UFBoundaryCriteria(StoppingCriteria):
    """
    Stop at:
      - a paragraph break (two or more newlines), or
      - a sentence-ending punctuation at buffer end,
    but do not stop immediately after a bare list marker like '1.' or 'A)' or 'Step 3:'.
    """

    def __init__(self, tokenizer, prefix_len: int, min_tokens: int = 8):
        self.tok = tokenizer
        self.prefix_len = prefix_len
        self.min_tokens = min_tokens

    def _newline_boundary(self, text: str) -> bool:
        if not text.strip():
            return False
        # Stop only on a paragraph break (two or more newlines at the end)
        if not re.search(r"\n{2,}$", text):
            return False
        # Drop trailing newlines to look at the last completed line.
        stripped = text.rstrip("\n")
        # If nothing before, do not stop.
        if not stripped:
            return False
        last_line = stripped.rsplit("\n", 1)[-1]
        # Do not stop if the last completed line is only a list marker.
        if _LIST_MARKER_LINE_RE.match(last_line):
            return False
        return True

    def _punct_boundary(self, text: str) -> bool:
        # Quick reject.
        if not _SENT_END_RE.search(text):
            return False
        # Do not stop on abbreviations at the very end.
        tail = text.rstrip()[-8:]  # small window is enough
        for ab in _ABBREV:
            if tail.endswith(ab):
                return False
        # Also do not stop if the *entire last line* is just a marker like "2." or "B)"
        last_line = text.rsplit("\n", 1)[-1].rstrip()
        if _LIST_MARKER_LINE_RE.match(last_line):
            return False
        return True

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the newly generated portion.
        gen_ids = input_ids[0, self.prefix_len :].tolist()
        if len(gen_ids) < self.min_tokens:
            return False
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        # Paragraph break has priority; otherwise punctuation boundary.
        return self._newline_boundary(text) or self._punct_boundary(text)


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    generator: torch.Generator = None,
) -> int:
    # logits: [1, vocab]
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1))

    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)

    # top-k filter
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, topk_idx, topk_vals)
        probs = filtered
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # top-p (nucleus) filter
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        # keep at least one token
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # sample without external RNG plumbing
    next_id = torch.multinomial(probs.squeeze(0), num_samples=1, generator=generator)
    return int(next_id.item())


def _is_boundary_like_generate(
    criteria: UFBoundaryCriteria, text: str, min_tokens: int, tok_ids_len: int
) -> bool:
    # mirror UFBoundaryCriteria.__call__: enforce min tokens, then newline or punct boundary
    if tok_ids_len < min_tokens:
        return False
    return criteria._newline_boundary(text) or criteria._punct_boundary(text)


# Deep-copy helpers for KV caches (tuple-of-tensors or HF cache objects)
def _clone_structure(x):
    """Recursively clone tensors and containers, preserving object type."""
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_structure(y) for y in x)
    if isinstance(x, dict):
        return {k: _clone_structure(v) for k, v in x.items()}
    # For cache objects (e.g., DynamicCache) or other custom classes
    if hasattr(x, "__dict__"):
        y = x.__class__.__new__(x.__class__)
        for k, v in x.__dict__.items():
            setattr(y, k, _clone_structure(v))
        return y
    # Last resort: shallow copy
    try:
        return copy.copy(x)
    except Exception:
        return x


def clone_past_kv(past_kv):
    """Deep copy HF past_key_values (tuple-of-tensors or cache object)."""
    try:
        return copy.deepcopy(past_kv)
    except Exception:
        return _clone_structure(past_kv)


# Import the MultiHeadCausalLM class (assuming it's available)
try:
    from multihead_model import MultiHeadCausalLM
except ImportError:
    print(
        "Warning: MultiHeadCausalLM not found. Please ensure multihead_model.py is available."
    )
    MultiHeadCausalLM = None


def download_from_hf_hub(
    hf_path: str, filename: str = "policy.pt", local_cache_dir: Optional[str] = None
) -> str:
    """
    Download a file from Hugging Face Hub to a local cache directory.

    Args:
        hf_path: Either a repo_id (like "Jennny/mdpo_llama8b_3heads_help_hon_truth1e6_1ep_bz192_cp15360")
                 or a full hf:// path (like "hf://Jennny/repo_name/policy.pt")
        filename: The filename to download (default: "policy.pt")
        local_cache_dir: Directory to store downloaded files, defaults to HF cache

    Returns:
        Path to locally cached file
    """
    # Parse the HF path to extract repo_id and filename
    if hf_path.startswith("hf://"):
        # Full path format: hf://username/repo_name/filename
        path_parts = hf_path[5:].split("/")  # Remove "hf://" prefix
        if len(path_parts) >= 3:
            repo_id = "/".join(path_parts[:2])  # username/repo_name
            filename = "/".join(path_parts[2:])  # everything after repo_name
        else:
            repo_id = "/".join(path_parts)
            # Keep default filename
    else:
        # Just repo_id format: username/repo_name
        repo_id = hf_path

    print(f"Downloading from Hugging Face Hub: {repo_id}/{filename}")

    try:
        # Download the file using huggingface_hub
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=local_cache_dir
        )

        print(f"Successfully downloaded from HF Hub to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading file from HF Hub: {e}")
        raise


def sanitize_model_name_for_filename(model_name: str) -> str:
    """
    Sanitize model name to be safe for use in filenames.
    """
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[/<>:"|?*\\]', "_", model_name)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized


def write_jsonl_result(file_path: str, result: dict):
    """
    Safely write a single result to JSONL file.
    Ensures proper single-line JSONL format.
    """
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            # Write as single line JSON (proper JSONL format)
            json_line = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            f.write(json_line + "\n")
            f.flush()  # Ensure data is written immediately
    except Exception as e:
        print(f"WARNING: Error writing to {file_path}: {e}")


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

            # Add common fallbacks
            tokenizer_sources.extend(
                [
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct",
                ]
            )

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
            logger.info("Added default chat template to tokenizer")

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

    def score_conversation(self, query: str, response: str) -> float:
        """
        Score a complete conversation using the unified reward model.
        Returns the probability of positive class (good response) as a float in [0, 1].
        """
        # Build conversation exactly like training data format
        conversation = f"User: {query}\n\nAssistant: {response}"

        input_ids = self.tokenizer.encode(conversation, return_tensors="pt").to(
            next(self.base_model.parameters()).device
        )
        attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        with torch.no_grad():
            # Get unified model prediction for conversation
            outputs = self.forward(input_ids, attention_mask)
            conversation_score = outputs.logits.squeeze().cpu().item()

        # Clean up tensors
        del input_ids, attention_mask
        torch.cuda.empty_cache()

        return conversation_score

    def score_multiple_conversations(
        self, query: str, responses: List[str]
    ) -> List[float]:
        """Score multiple conversations efficiently."""
        if not responses:
            return []

        scores = []
        for response in responses:
            score = self.score_conversation(query, response)
            scores.append(score)

        return scores


class UnifiedGuidedSentenceLevelDecoder:
    def __init__(
        self,
        model_name: str = "Jennny/mdpo_llama8b_3heads_help_hon_truth1e6_1ep_bz192_cp15360",
        base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        unified_reward_model_name: str = "Jennny/unified_rm_1e5_1600",
        max_sentences: int = 100,
        max_tokens_per_sentence: int = 256,
        max_total_tokens: int = 1024,
        chunk_size: int = 256,
        num_candidates: int = 5,
        num_heads: int = 3,
        use_ensemble: bool = True,
        head_index: Optional[int] = None,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.8,
        dtype: str = "bfloat16",
        cache_dir: Optional[str] = None,
        unified_device: str = "cuda:0",
        use_base_model_only: bool = False,
        random_seed: Optional[int] = None,
    ):
        # Store configuration
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.unified_reward_model_name = unified_reward_model_name
        self.max_sentences = max_sentences
        self.max_tokens_per_sentence = max_tokens_per_sentence
        self.max_total_tokens = max_total_tokens
        self.chunk_size = chunk_size
        self.num_candidates = num_candidates
        self.num_heads = num_heads
        self.use_ensemble = use_ensemble
        self.head_index = head_index
        self.use_base_model_only = use_base_model_only

        # Initialize unified reward model
        logger.info(f"Loading unified reward model on {unified_device}...")
        try:
            self.unified_reward_model = UnifiedRewardModel.from_pretrained(
                unified_reward_model_name,
                base_model_name=base_model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": unified_device},
                trust_remote_code=True,
            ).eval()
            logger.info(
                f"Successfully loaded unified reward model: {unified_reward_model_name}"
            )
        except Exception as e:
            logger.error(f"ERROR: Failed to load unified reward model: {e}")
            raise

        # Initialize random state
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
        self.random_seed = random_seed
        self.sample_counter = 0  # Counter for generating unique seeds

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # Use only <|eot_id|> as EOS for Llama chat and set pad_token_id accordingly
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.eot_id is None:
            # Fallback if tokenizer does not know the token
            self.eot_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.eot_id

        # Load the model (either multi-head DPO or base model)
        if self.use_base_model_only:
            self.model = self._load_base_model(dtype, cache_dir)
        else:
            self.model = self._load_multi_head_model(dtype, cache_dir)

        print(f"EOS/EOT token id: {self.eot_id}")

    def _load_multi_head_model(self, dtype: str, cache_dir: Optional[str] = None):
        """
        Load the multi-head DPO model from Hugging Face Hub.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)

        print(f"Loading base model for multi-head DPO: {self.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # Create MultiHeadCausalLM model
        if MultiHeadCausalLM is None:
            raise ImportError(
                "MultiHeadCausalLM not available. Please ensure multihead_model.py is in the path."
            )

        print(f"Creating multi-head model with {self.num_heads} heads")
        model = MultiHeadCausalLM(base_model, num_heads=self.num_heads).to(device)

        # Download and load the checkpoint from HF Hub
        print(f"Loading multi-head DPO checkpoint from HF Hub: {self.model_name}")
        local_checkpoint_path = download_from_hf_hub(
            self.model_name, filename="policy.pt", local_cache_dir=cache_dir
        )

        print(f"Loading checkpoint: {local_checkpoint_path}")
        checkpoint = torch.load(local_checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state" in checkpoint:
            state_dict = checkpoint["state"]
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()
        return model

    def _load_base_model(self, dtype: str, cache_dir: Optional[str] = None):
        """
        Load the base model from Hugging Face Hub.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)

        print(f"Loading base model: {self.base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        return model

    def generate_candidate_chunks(
        self, query: str, running_text: str, step_idx: int
    ) -> List[Tuple[str, bool, str]]:
        """
        Generate multiple candidate chunks for unified reward evaluation with independent RNG streams.
        Returns a list of (chunk, is_eos, finish_reason).
        """
        candidates: List[Tuple[str, bool, str]] = []
        max_attempts = self.num_candidates * 3
        attempts = 0
        eos_only_count = 0
        while len(candidates) < self.num_candidates and attempts < max_attempts:
            attempts += 1
            chunk, is_eos, finish_reason = self.generate_single_sentence(
                query, running_text
            )

            if chunk:
                candidates.append((chunk, is_eos, finish_reason))
            if not chunk and is_eos:
                eos_only_count += 1

        print(f"   Attempts: {attempts}, EOS only count: {eos_only_count}")
        if eos_only_count == attempts:
            return [("", True, "eos_only")]

        return candidates

    def generate_single_sentence(
        self, query: str, running_text: str
    ) -> Tuple[str, bool, str]:
        """
        Generate a single contiguous chunk using UF-aware boundary stopping.
        Returns (chunk, is_eos, finish_reason).
        """
        # Build the conversation history
        messages = [{"role": "user", "content": query}]
        if running_text:
            messages.append({"role": "assistant", "content": running_text})

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=bool(running_text),
            add_generation_prompt=not running_text,
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )

        # UF-aware stopping criteria that halts at the first boundary
        prefix_len = inputs["input_ids"].shape[1]
        stops = StoppingCriteriaList(
            [UFBoundaryCriteria(self.tokenizer, prefix_len, min_tokens=8)]
        )

        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": self.max_tokens_per_sentence,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": self.eot_id,
            "eos_token_id": self.eot_id,
            "stopping_criteria": stops,
            "use_cache": True,
        }

        # Generate
        with torch.no_grad():
            if self.use_base_model_only:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs,
                )
            elif self.use_ensemble:
                try:
                    outputs = self.model.generate_ensemble(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_kwargs,
                    )
                except AttributeError:
                    outputs = self.model.generate_with_head(
                        0,
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_kwargs,
                    )
            elif self.head_index is not None:
                outputs = self.model.generate_with_head(
                    self.head_index,
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs,
                )
            else:
                outputs = self.model.generate_with_head(
                    0,
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs,
                )

        # Extract the newly generated slice
        generated_tokens = outputs[0][len(inputs["input_ids"][0]) :]
        chunk = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Check EOS (<|eot_id|>)
        has_eos = any(token_id == self.eot_id for token_id in generated_tokens.tolist())
        if has_eos and not chunk.strip():
            return ("", True, "eos_token")

        # Cleanup
        del outputs, formatted_prompt, messages, inputs
        torch.cuda.empty_cache()

        return chunk, bool(has_eos), ("eos_token" if has_eos else "stop")

    def select_best_candidate_with_unified_reward(
        self,
        query: str,
        running_text: str,
        candidates: List[Tuple[str, bool, str]],
    ) -> Tuple[str, bool, str, float]:
        """
        Select the best candidate chunk using unified reward scoring of running_text + chunk.
        Returns (best_chunk, best_is_eos, best_finish_reason, best_score).
        """
        if not candidates:
            return "", False, "no_candidates", 0.0

        rollouts = []
        for chunk, is_eos, finish_reason in candidates:
            rollouts.append(_normalize_join(running_text, chunk))

        scores = self.unified_reward_model.score_multiple_conversations(query, rollouts)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_chunk, best_is_eos, best_finish_reason = candidates[best_idx]
        best_score = scores[best_idx]
        return best_chunk, best_is_eos, best_finish_reason, best_score

    def decode_with_hidden_state(
        self,
        query: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        min_tokens_per_chunk: int = 8,
    ) -> Tuple[str, List[str], Dict]:
        """
        Single continuous decode driven by past_key_values.
        Emits chunks at UF boundary without rebuilding the prompt each time.
        Supports multi-candidate generation with unified reward selection per step.
        """
        print("\n================ QUERY ================\n")
        print(query)
        print("\n======================================\n")

        # 1) Build the chat prompt ONCE
        messages = [{"role": "user", "content": query}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(formatted, return_tensors="pt").to(device)

        # 2) Initial forward pass to seed KV cache and logits
        with torch.no_grad():
            if self.use_base_model_only:
                out = self.model(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Use multi-head forward with either ensemble or a specific head
                out = self.model(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    use_cache=True,
                    return_dict=True,
                    head_index=(
                        None
                        if self.use_ensemble
                        else (self.head_index if self.head_index is not None else 0)
                    ),
                )
        main_past_kv = out.past_key_values
        main_next_logits = out.logits[:, -1, :]

        # 3) Bookkeeping
        chunks_log: List[str] = []
        running_text = ""
        total_tokens = 0
        emitted_chunks = 0
        stop_reason = "unknown"
        eos_hit = False
        unified_scores: List[float] = []

        # UF stopping helper
        uf_criteria = UFBoundaryCriteria(
            self.tokenizer,
            prefix_len=inputs["input_ids"].shape[1],
            min_tokens=min_tokens_per_chunk,
        )

        # Helper to sample a single candidate starting from a given state
        def sample_candidate(
            start_past_kv,
            start_logits,
        ) -> Tuple[str, bool, str, int, tuple, torch.Tensor, List[int]]:
            cur_chunk_ids: List[int] = []
            # Deep copy the cache so each candidate mutates its own copy
            local_past = clone_past_kv(start_past_kv)
            # logits tensor can be safely cloned to avoid in-place ops
            local_logits = start_logits.detach().clone()
            local_tokens = 0

            candidate_seed = (self.random_seed or 0) + (
                self.sample_counter * 10007
            )  # Large prime
            generator = torch.Generator(device=local_logits.device)
            generator.manual_seed(candidate_seed)
            self.sample_counter += 1

            while True:
                # Respect per-step and total limits while proposing a candidate
                if local_tokens >= self.max_tokens_per_sentence:
                    finish_reason = "max_tokens_per_sentence"
                    break
                if total_tokens + local_tokens >= self.max_total_tokens:
                    finish_reason = "total_token_limit"
                    break

                next_id = _sample_next_token(
                    local_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=generator,  # Use the unique generator for this candidate
                )

                # Append
                cur_chunk_ids.append(next_id)
                local_tokens += 1

                # EOS check
                if next_id == self.eot_id or next_id == self.tokenizer.eos_token_id:
                    # Return whatever was generated (possibly empty)
                    text = self.tokenizer.decode(
                        cur_chunk_ids, skip_special_tokens=True
                    )
                    return (
                        text,
                        True,
                        "eos_token",
                        local_tokens,
                        local_past,
                        local_logits,
                        cur_chunk_ids,
                    )

                # Advance hidden state
                with torch.no_grad():
                    if self.use_base_model_only:
                        out_step = self.model(
                            input_ids=torch.tensor(
                                [[next_id]], device=device, dtype=torch.long
                            ),
                            past_key_values=local_past,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        out_step = self.model(
                            input_ids=torch.tensor(
                                [[next_id]], device=device, dtype=torch.long
                            ),
                            past_key_values=local_past,
                            use_cache=True,
                            return_dict=True,
                            head_index=(
                                None
                                if self.use_ensemble
                                else (
                                    self.head_index
                                    if self.head_index is not None
                                    else 0
                                )
                            ),
                        )
                local_past = out_step.past_key_values
                local_logits = out_step.logits[:, -1, :]

                # Boundary check
                cur_text = self.tokenizer.decode(
                    cur_chunk_ids, skip_special_tokens=True
                )
                if _is_boundary_like_generate(
                    uf_criteria, cur_text, min_tokens_per_chunk, len(cur_chunk_ids)
                ):
                    finish_reason = "boundary"
                    break

            # Emit candidate text
            text = self.tokenizer.decode(cur_chunk_ids, skip_special_tokens=True)
            return (
                text,
                False,
                finish_reason,
                local_tokens,
                local_past,
                local_logits,
                cur_chunk_ids,
            )

        # 4) Step loop
        step_idx = 0

        while True:
            if total_tokens >= self.max_total_tokens:
                stop_reason = "total_token_limit"
                break
            if emitted_chunks >= self.max_sentences:
                stop_reason = "sentence_count_limit"
                break

            # Generate candidates
            candidates_raw = (
                []
            )  # (text, is_eos, reason, tokens, past, logits, token_ids)
            max_attempts = max(self.num_candidates, 1) * 3
            attempts = 0
            eos_only_count = 0
            while (
                len(candidates_raw) < max(self.num_candidates, 1)
                and attempts < max_attempts
            ):
                attempts += 1
                text, is_eos, reason, cand_tokens, cand_past, cand_logits, cand_ids = (
                    sample_candidate(main_past_kv, main_next_logits)
                )
                # Only keep non-empty text; track eos-only to decide stop
                if text:
                    candidates_raw.append(
                        (
                            text,
                            is_eos,
                            reason,
                            cand_tokens,
                            cand_past,
                            cand_logits,
                            cand_ids,
                        )
                    )
                elif is_eos:
                    # Record eos-only as a sentinel if nothing else is collected
                    candidates_raw.append(
                        ("", True, "eos_token", 0, cand_past, cand_logits, cand_ids)
                    )
                    eos_only_count += 1

            if eos_only_count == len(candidates_raw):
                stop_reason = "eos_token"
                eos_hit = True
                break

            if not candidates_raw:
                stop_reason = "no_valid_candidates"
                break

            # Choose candidate
            if self.num_candidates == 1:
                best_idx = 0
                best_score = float("nan")
            else:
                rollouts = [running_text + c[0] for c in candidates_raw]
                scores = self.unified_reward_model.score_multiple_conversations(
                    query, rollouts
                )

                print(f"\n--- Step {step_idx} candidates and scores ---\n")
                for i, (cand_text, _, _, _, _, _, _) in enumerate(candidates_raw):
                    preview = cand_text.replace("\n", "\\n")
                    print(f"[{i}] score={scores[i]:.4f}")
                    print(preview)
                    print()

                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                best_score = scores[best_idx]

            (
                best_text,
                best_is_eos,
                best_reason,
                best_tokens,
                best_past,
                best_logits,
                best_ids,
            ) = candidates_raw[best_idx]

            # Apply chosen candidate to main state
            if best_text:
                print(
                    f"Step {step_idx} chosen, absolute_score={best_score if best_score==best_score else float('nan'):.4f}, reason={best_reason}"
                )
                print(best_text)
                running_text += best_text
                print(f"Current running text at step {step_idx}:")
                print(running_text)
                chunks_log.append(best_text)
                total_tokens += best_tokens
                emitted_chunks += 1
                if not (best_score != best_score):  # not NaN
                    unified_scores.append(best_score)

            if best_is_eos or best_reason == "eos_token":
                stop_reason = "eos_token"
                eos_hit = True
                break

            # Update main KV/logits for continued decoding
            main_past_kv = best_past
            main_next_logits = best_logits
            step_idx += 1

            # Check limits after applying chunk
            if total_tokens >= self.max_total_tokens:
                stop_reason = "total_token_limit"
                break
            if emitted_chunks >= self.max_sentences:
                stop_reason = "sentence_count_limit"
                break

        metadata = {
            "total_sentences": len(chunks_log),
            "total_tokens": total_tokens,
            "stop_reason": stop_reason,
            "eos_encountered": eos_hit,
            "unified_scores": unified_scores,
            "model_name": self.model_name,
            "base_model_name": self.base_model_name,
            "unified_reward_model_name": self.unified_reward_model_name,
            "num_candidates": self.num_candidates,
            "num_heads": self.num_heads,
            "use_ensemble": self.use_ensemble,
            "head_index": self.head_index,
            "use_base_model_only": self.use_base_model_only,
        }

        print(f"\n💡 Final response:")
        print(running_text)

        return running_text, chunks_log, metadata


def get_sample_queries_with_splits(
    total_samples: int = 500, total_splits: int = 5, split_id: int = 0, seed: int = 42
) -> List[str]:
    """
    Sample queries from UltraFeedback dataset and return the specified split.

    Args:
        total_samples: Total number of samples to draw from dataset
        total_splits: Total number of splits to divide samples into
        split_id: Which split to return (0-indexed)
        seed: Random seed for reproducibility

    Returns:
        List of queries for the specified split
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Load the dataset
    dataset = load_dataset("openbmb/UltraFeedback", split="train")

    # Sample the total number of queries with fixed seed
    all_queries = random.sample(dataset["instruction"], total_samples)

    # Calculate split size
    split_size = total_samples // total_splits
    remainder = total_samples % total_splits

    # Calculate start and end indices for this split
    start_idx = split_id * split_size
    if split_id < remainder:
        # First 'remainder' splits get one extra sample
        start_idx += split_id
        end_idx = start_idx + split_size + 1
    else:
        # Later splits use standard size
        start_idx += remainder
        end_idx = start_idx + split_size

    # Extract the split
    split_queries = all_queries[start_idx:end_idx]

    print(
        f"Split {split_id}/{total_splits-1}: {len(split_queries)} queries (indices {start_idx}-{end_idx-1})"
    )

    return split_queries


def generate_output_filename(
    model_name: str,
    split_id: int = None,
    total_splits: int = None,
    suffix: str = "",
) -> str:
    """
    Generate a descriptive output filename based on the model name.
    """
    # Sanitize model name for filename
    sanitized_model = sanitize_model_name_for_filename(model_name)

    # Build filename components
    filename_parts = ["unified_guided_results"]

    # Add sanitized model name
    filename_parts.append(sanitized_model)

    # Add split information if provided
    if split_id is not None and total_splits is not None:
        filename_parts.append(f"split_{split_id}_of_{total_splits}")

    # Add suffix if provided
    if suffix:
        filename_parts.append(suffix)

    # Join with underscores and add extension
    filename = "_".join(filename_parts) + ".jsonl"

    return filename


def test_unified_guided_decoder(
    queries: List[str] = None,
    output_file: str = None,
    original_responses: List[Optional[str]] = None,
    model_name: str = "Jennny/mdpo_llama8b_3heads_help_hon_truth1e6_1ep_bz192_cp15360",
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    unified_reward_model_name: str = "Jennny/unified_rm_1e5_1600",
    num_candidates: int = 5,
    num_heads: int = 3,
    use_ensemble: bool = True,
    head_index: Optional[int] = None,
    dtype: str = "bfloat16",
    cache_dir: Optional[str] = None,
    unified_device: str = "cuda:0",
    max_total_tokens: int = 1024,
    max_tokens_per_sentence: int = 256,
    max_sentences: int = 100,
    use_base_model_only: bool = False,
    random_seed: Optional[int] = None,
):
    """
    Test the UnifiedGuidedSentenceLevelDecoder on sample queries.
    """
    if queries is None:
        queries = [
            "Write a detailed essay about ancient type of religion, totemism",
            "Hi there! I'm becoming a life coach, and I need help getting started. For my research, I would like you to act as a life coach so I have some direction. Could you do that for me?",
            "Explain the concept of artificial intelligence and its applications in modern society",
            "What are the key principles of sustainable living?",
            "Describe the process of photosynthesis in plants",
        ]

    # Generate default output filename if not provided
    if output_file is None:
        output_file = generate_output_filename(model_name, suffix="test")

    print(f"🔍 Testing {len(queries)} queries with unified reward guidance")
    print(f"Output file: {output_file}")
    print(f"🤖 Model: {model_name}")
    print(f"Base Model: {base_model_name}")

    if original_responses is None:
        original_responses = [None] * len(queries)

    if random_seed is None:
        import time

        random_seed = int(time.time())

    # Initialize the decoder
    decoder = UnifiedGuidedSentenceLevelDecoder(
        model_name=model_name,
        base_model_name=base_model_name,
        unified_reward_model_name=unified_reward_model_name,
        num_candidates=num_candidates,
        num_heads=num_heads,
        use_ensemble=use_ensemble,
        head_index=head_index,
        dtype=dtype,
        cache_dir=cache_dir,
        unified_device=unified_device,
        max_total_tokens=max_total_tokens,
        max_tokens_per_sentence=max_tokens_per_sentence,
        max_sentences=max_sentences,
        use_base_model_only=use_base_model_only,
        random_seed=random_seed,
    )

    # Clear output file
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    results = []

    for i, query in enumerate(queries):
        print(f"\n\n{'='*80}")
        print(f"Testing query {i+1}/{len(queries)}")
        print(f"{'='*80}\n")

        try:
            # Generate response
            full_response, sentences, metadata = decoder.decode_with_hidden_state(
                query, temperature=1.0, top_p=1.0, top_k=50
            )

            orig_resp = original_responses[i]
            result = {
                "query_id": i,
                "query": query,
                "original_response": orig_resp,
                "full_response": full_response,
                "sentences": sentences,
                "metadata": metadata,
            }

            results.append(result)

            # Save result using safe JSONL writing
            write_jsonl_result(output_file, result)

            print(f"\nResult saved to {output_file}")

        except Exception as e:
            print(f"ERROR: Error processing query: {e}")
            error_result = {
                "query_id": i,
                "query": query,
                "error": str(e),
                "full_response": "",
                "sentences": [],
                "metadata": {"error": str(e)},
            }

            write_jsonl_result(output_file, error_result)

        # Cleanup after each query
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n🎉 Testing complete! Processed {len(queries)} queries.")
    print(f"Results saved to: {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total_samples",
        type=int,
        default=500,
        help="Total number of samples to draw from dataset",
    )
    parser.add_argument(
        "--unified_device",
        type=str,
        default="cuda:0",
        help="Device to use for the unified reward model (default: cuda:0)",
    )
    parser.add_argument(
        "--total_splits",
        type=int,
        default=5,
        help="Total number of splits to divide samples into",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=0,
        help="Which split to process (0-indexed, must be < total_splits)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Jennny/mdpo_llama8b_3heads_help_hon_truth1e6_1ep_bz192_cp15360",
        help="Which multi-head DPO model to use for generation",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Which base model architecture to use",
    )
    parser.add_argument(
        "--unified_reward_model_name",
        type=str,
        default="Jennny/unified_rm_1e5_1600",
        help="Which unified reward model to use for scoring",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=5,
        help="Number of candidate sentences to generate at each step",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=3,
        help="Number of heads in the multi-head model",
    )
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        default=True,
        help="Use ensemble generation across all heads",
    )
    parser.add_argument(
        "--head_index",
        type=int,
        default=None,
        help="Specific head to use (overrides ensemble)",
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
        default=None,
        help="Directory to cache downloaded models",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to write outputs (will auto-generate if not provided)",
    )
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=1024,
        help="Maximum total tokens for the complete response",
    )
    parser.add_argument(
        "--max_tokens_per_sentence",
        type=int,
        default=256,
        help="Maximum tokens per individual sentence",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=100,
        help="Maximum number of sentences to generate",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_base_model_only",
        action="store_true",
        default=False,
        help="Only use the base model for generation, skipping multi-head DPO",
    )
    args = parser.parse_args()

    # Validate split_id
    if args.split_id >= args.total_splits or args.split_id < 0:
        raise ValueError(
            f"split_id ({args.split_id}) must be >= 0 and < total_splits ({args.total_splits})"
        )

    # Generate output filename if not provided
    if args.output_file is None:
        args.output_file = generate_output_filename(
            args.model_name,
            split_id=args.split_id,
            total_splits=args.total_splits,
        )

    # Determine queries from the specified split
    queries = get_sample_queries_with_splits(
        total_samples=args.total_samples,
        total_splits=args.total_splits,
        split_id=args.split_id,
    )

    print(
        f"Processing split {args.split_id}/{args.total_splits-1} with {len(queries)} queries"
    )
    print(f"🤖 Using Multi-Head DPO Model: {args.model_name}")
    print(f"Base Model: {args.base_model_name}")
    print(f"🏆 Using Unified Reward Model: {args.unified_reward_model_name}")
    print(f"🎲 Generating {args.num_candidates} candidates per sentence")
    print(f"Output file: {args.output_file}")
    print(f"Ensemble: {args.use_ensemble}, Head: {args.head_index}")
    print(f"🌱 Base Model Only: {args.use_base_model_only}")

    test_unified_guided_decoder(
        queries=queries,
        output_file=args.output_file,
        model_name=args.model_name,
        base_model_name=args.base_model_name,
        unified_reward_model_name=args.unified_reward_model_name,
        num_candidates=args.num_candidates,
        num_heads=args.num_heads,
        use_ensemble=args.use_ensemble,
        head_index=args.head_index,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        unified_device=args.unified_device,
        max_total_tokens=args.max_total_tokens,
        max_tokens_per_sentence=args.max_tokens_per_sentence,
        max_sentences=args.max_sentences,
        use_base_model_only=args.use_base_model_only,
        random_seed=args.random_seed,
    )
