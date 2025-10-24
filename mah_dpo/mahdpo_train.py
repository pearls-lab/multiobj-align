########################
# This script is modified from DPO codebase https://github.com/eric-mitchell/direct-preference-optimization/blob/main/train.py
########################
import os

# Optionally help avoid CUDA fragmentation issues.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import (
    get_local_dir,
    get_local_run_dir,
    disable_dropout,
    init_distributed,
    get_open_port,
)
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import mahdpo_trainers
import wandb
import json
import socket
from typing import Optional, Set, List
import resource
from multihead_model import MultiHeadCausalLM

# import boto3
from urllib.parse import urlparse
import tempfile

from huggingface_hub import hf_hub_download
import re

OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
)


def download_from_s3(s3_path, local_path=None):
    """Download a file from S3 to a local path."""
    # Parse the S3 path
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    # If no local path is provided, create a temporary directory
    if local_path is None:
        os.makedirs(os.path.join(tempfile.gettempdir(), "s3_downloads"), exist_ok=True)
        local_path = os.path.join(
            tempfile.gettempdir(), "s3_downloads", os.path.basename(key)
        )
    else:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Create a session and s3 client
    session = boto3.Session()
    s3 = session.client("s3")

    # Download the file
    print(f"Downloading from S3: {s3_path} to {local_path}")
    s3.download_file(bucket, key, local_path)
    return local_path


def download_from_hf_hub(hf_path, filename="policy.pt", local_path=None):
    """
    Download a file from the Hugging Face Hub and return the local path.
    Supports both hf://user/repo/file.pt, user/repo, and user/repo/file.pt formats.
    """
    if hf_path.startswith("hf://"):
        parts = hf_path[5:].split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:]) if len(parts) > 2 else filename
    else:
        # treat plain user/repo or user/repo/file.pt
        parts = hf_path.split("/")
        repo_id = "/".join(parts[:2])
        if len(parts) > 2:
            filename = "/".join(parts[2:])

    if local_path is None:
        local_path = os.path.join(tempfile.gettempdir(), "hf_downloads")

    os.makedirs(local_path, exist_ok=True)
    print(f"Downloading from Hugging Face Hub: {repo_id}/{filename}")
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=local_path)


def is_hf_hub_path(path: str) -> bool:
    """Return True when *path* looks like an HF repo or the hf:// scheme."""
    if path.startswith("hf://"):
        return True
    # user/repo or user/repo/file.ext, but not local paths or S3
    if path.startswith("s3://") or path.startswith("./") or path.startswith("/"):
        return False
    return "/" in path and not os.path.exists(path)


def worker_main(
    rank: int,
    world_size: int,
    config: DictConfig,
    policy: nn.Module,
    reference_model: Optional[nn.Module] = None,
):
    """Main function for each worker process."""
    # For FSDP workers, set the CUDA device.
    if "FSDP" in config.trainer:
        torch.cuda.set_device(rank)
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ["WANDB_CACHE_DIR"] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(mahdpo_trainers, config.trainer)
    print(f"Creating trainer on process {rank} with world size {world_size}")
    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
    )
    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training."""
    OmegaConf.resolve(config)

    # Check for missing configuration keys
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    # Validate evaluation settings
    if config.eval_every % config.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    # Setup FSDP port if needed
    if "FSDP" in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print("No FSDP port specified; using open port:", free_port)
        config.fsdp_port = free_port

    # Log the complete configuration
    print(OmegaConf.to_yaml(config))

    # Write config to disk
    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    # Setup cache directory for model downloads
    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)
    print("Building policy")

    # Model loading options - no device map for proper FSDP support
    model_kwargs = {}

    # Get the requested policy dtype
    policy_dtype = getattr(torch, config.model.policy_dtype)
    print(f"Using policy dtype: {policy_dtype}")

    # Load pretrained model
    print(f"Loading model from {config.model.name_or_path}")
    base_policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True,
        torch_dtype=policy_dtype,  # Set dtype during model loading
        **model_kwargs,
    )
    disable_dropout(base_policy)

    # CRITICAL: Print model device before wrapping
    print(f"Base policy device after loading: {next(base_policy.parameters()).device}")
    print(f"Base policy dtype after loading: {next(base_policy.parameters()).dtype}")

    reference_model = None
    if config.loss.name in {"dpo", "ipo"}:
        print("Building reference model with UNPERTURBED heads")

        if hasattr(config.model, "num_heads") and config.model.num_heads > 1:
            # Load a separate base model for reference
            reference_model_dtype = getattr(torch, config.model.reference_dtype)
            print(f"Using reference model dtype: {reference_model_dtype}")

            base_reference = transformers.AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path,
                cache_dir=get_local_dir(config.local_dirs),
                low_cpu_mem_usage=True,
                torch_dtype=reference_model_dtype,
                **model_kwargs,
            )
            disable_dropout(base_reference)

            # Create reference model with unperturbed heads
            reference_model = MultiHeadCausalLM(
                base_reference,
                num_heads=config.model.num_heads,
                equal_init=False,
            )

            # Freeze reference model parameters
            for param in reference_model.parameters():
                param.requires_grad = False
            print(
                "Created reference model with unperturbed SFT heads and frozen parameters"
            )
        else:
            # Single head case - use standard approach
            reference_model_dtype = getattr(torch, config.model.reference_dtype)
            print(f"Using reference model dtype: {reference_model_dtype}")

            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path,
                cache_dir=get_local_dir(config.local_dirs),
                low_cpu_mem_usage=True,
                torch_dtype=reference_model_dtype,
                **model_kwargs,
            )
            disable_dropout(reference_model)

    # Create policy model with perturbed heads for multi-head models
    if hasattr(config.model, "num_heads") and config.model.num_heads > 1:
        policy = MultiHeadCausalLM(
            base_policy,
            num_heads=config.model.num_heads,
            equal_init=True,
        )
        print("Created policy model with perturbed heads")
    else:
        policy = base_policy

    # Keep model on CPU when using FSDP (handle device placement in the trainer)
    if "FSDP" in config.trainer:
        device = torch.device("cpu")
        print("Keeping models on CPU for FSDP initialization")
    else:
        # For other trainers, move to GPU immediately
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device)
        if reference_model is not None:
            reference_model = reference_model.to(device)
        print(f"Moved models to {device}")

    # Load checkpoint if specified
    if config.model.archive is not None:
        archive_path = config.model.archive

        # Check if the archive path is a Hugging Face Hub path
        if is_hf_hub_path(archive_path):
            local_dir = os.path.join(
                get_local_dir(config.local_dirs), "checkpoint_downloads"
            )
            os.makedirs(local_dir, exist_ok=True)
            archive_path = download_from_hf_hub(archive_path, local_path=local_dir)

        # Check if the archive path is an S3 URL
        elif archive_path.startswith("s3://"):
            # Download from S3 to local cache
            local_dir = os.path.join(
                get_local_dir(config.local_dirs), "checkpoint_downloads"
            )
            os.makedirs(local_dir, exist_ok=True)
            local_archive_path = os.path.join(local_dir, os.path.basename(archive_path))
            archive_path = download_from_s3(archive_path, local_archive_path)

        print(f"Loading checkpoint from {archive_path}")
        checkpoint = torch.load(archive_path, map_location="cpu")
        step, metrics = checkpoint["step_idx"], checkpoint["metrics"]
        print(
            f"Loading pre-trained weights at step {step} with metrics {json.dumps(metrics, indent=2)}"
        )

        state = checkpoint["state"]

        # Load state dict into policy model
        try:
            # For multi-head models, try loading the state dict directly first
            policy.load_state_dict(state, strict=False)
            print("Loaded pre-trained weights into policy model")
        except Exception as e:
            print(f"Direct loading failed: {e}")
            print("Trying with key remapping...")

            # Fallback: try key remapping for compatibility
            new_state = {}
            for key, value in state.items():
                new_key = key if key.startswith("base_model.") else "base_model." + key
                new_state[new_key] = value

            try:
                policy.load_state_dict(new_state, strict=False)
                print("Loaded pre-trained weights into policy model with remapping")
            except Exception as e2:
                print(f"Warning: Error loading checkpoint even with remapping: {e2}")
                print("Continuing with training from base model")

        # Load checkpoint into reference model to maintain consistent SFT state
        if config.loss.name in {"dpo", "ipo"} and reference_model is not None:
            try:
                # Try direct loading first
                reference_model.load_state_dict(state, strict=False)
                print("Loaded pre-trained weights into reference model")
            except Exception as e:
                print(f"Reference direct loading failed: {e}, trying remapping...")
                try:
                    # Try with key remapping
                    new_state = {}
                    for key, value in state.items():
                        new_key = (
                            key
                            if key.startswith("base_model.")
                            else "base_model." + key
                        )
                        new_state[new_key] = value
                    reference_model.load_state_dict(new_state, strict=False)
                    print(
                        "Loaded pre-trained weights into reference model with remapping"
                    )
                except Exception as e2:
                    print(f"Warning: Error loading checkpoint into reference: {e2}")
                    print("Reference model will use base weights")

            # Ensure reference model remains frozen after loading
            for param in reference_model.parameters():
                param.requires_grad = False
            print("Re-frozen reference model parameters after checkpoint loading")

    # Start worker process or run single-process
    if "FSDP" in config.trainer:
        world_size = torch.cuda.device_count()
        print("Starting", world_size, "processes for FSDP training")

        # Increase file limit for FSDP
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Setting RLIMIT_NOFILE soft limit to {hard} from {soft}")

        # Launch distributed processes
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy, reference_model),
            join=True,
        )
    else:
        print("Starting single-process worker")
        worker_main(0, 1, config, policy, reference_model)


if __name__ == "__main__":
    main()
