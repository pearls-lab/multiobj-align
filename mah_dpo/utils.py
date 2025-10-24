########################
# This script is adopted from DPO codebase https://github.com/eric-mitchell/direct-preference-optimization/blob/main/utils.py
########################
import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # bind to all interfaces and use an OS provided port
        return s.getsockname()[1]  # return only the port number


def get_remote_file(remote_path, local_path=None):
    hostname, path = remote_path.split(":")
    local_hostname = socket.gethostname()
    if (
        hostname == local_hostname
        or hostname == local_hostname[: local_hostname.find(".")]
    ):
        return path

    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Copying {hostname}:{path} to {local_path}")
    os.system(f"scp {remote_path} {local_path}")
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"


def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


import torch


def slice_and_move_batch_for_device(
    batch, microbatch_idx, total_microbatches, device_idx
):
    # Try to determine batch size (B) from one of the tensor entries.
    B = None
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            B = v.size(0)
            break
        elif isinstance(v, list):
            # If it's a list, we assume its length is the batch size.
            B = len(v)
            break
    if B is None:
        raise ValueError(
            "No tensor or list with a length was found in the batch to determine batch size."
        )

    slice_size = B // total_microbatches
    start = microbatch_idx * slice_size
    # For the last microbatch, take all remaining examples.
    end = start + slice_size if microbatch_idx < total_microbatches - 1 else B
    # If the computed slice would be empty, default to the entire batch
    if start >= end:
        start, end = 0, B

    device = torch.device("cuda", device_idx)
    sliced_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            sliced_batch[k] = v[start:end].to(device)
        elif isinstance(v, list):
            # If the list's length matches the batch size, slice it.
            if len(v) == B:
                sliced_batch[k] = v[start:end]
            else:
                # Otherwise, leave it unchanged.
                sliced_batch[k] = v
        else:
            # For other types, leave them unchanged.
            sliced_batch[k] = v
    return sliced_batch


# Add to utils.py
def slice_and_move_batch_for_device_balanced(
    batch, microbatch_idx, total_microbatches, device_idx, preserve_dimensions=True
):
    """
    Slice batch for device while preserving dimension balance.

    When preserve_dimensions is True, ensures each device gets a balanced mix of examples from different dimensions.
    """
    # Try to determine batch size (B) from one of the tensor entries.
    B = None
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            B = v.size(0)
            break
        elif isinstance(v, list):
            # If it's a list, we assume its length is the batch size.
            B = len(v)
            break
    if B is None:
        raise ValueError(
            "No tensor or list with a length was found in the batch to determine batch size."
        )

    # Check if we need to preserve dimension balance
    if preserve_dimensions and "dimensions" in batch:
        dimensions = batch["dimensions"]
        num_dims = max(dimensions) + 1 if dimensions else 1

        # Create indices for each dimension
        dim_indices = [[] for _ in range(num_dims)]
        for i, dim in enumerate(dimensions):
            if dim < num_dims:
                dim_indices[dim].append(i)

        # Calculate examples per dimension per microbatch
        sliced_indices = []
        for dim_idx in range(num_dims):
            dim_size = len(dim_indices[dim_idx])
            if dim_size == 0:
                continue

            # Calculate slice size for this dimension
            dim_slice_size = dim_size // total_microbatches
            if dim_slice_size == 0:
                # If too small, give at least one example per dimension
                dim_slice_size = 1

            # Calculate start and end indices for this dimension
            dim_start = microbatch_idx * dim_slice_size
            dim_end = (
                dim_start + dim_slice_size
                if microbatch_idx < total_microbatches - 1
                else dim_size
            )

            # Add indices for this dimension to our selection
            if dim_start < dim_end:
                sliced_indices.extend(dim_indices[dim_idx][dim_start:dim_end])

        # If we end up with an empty slice somehow, take one element from each dimension
        if not sliced_indices and any(dim_indices):
            for dim_idx in range(num_dims):
                if dim_indices[dim_idx]:
                    sliced_indices.append(dim_indices[dim_idx][0])

        # Sort indices to maintain original ordering as much as possible
        sliced_indices.sort()
    else:
        # Standard slicing without dimension preservation
        slice_size = B // total_microbatches
        start = microbatch_idx * slice_size
        # For the last microbatch, take all remaining examples.
        end = start + slice_size if microbatch_idx < total_microbatches - 1 else B
        # If the computed slice would be empty, default to the entire batch
        if start >= end:
            start, end = 0, B

        sliced_indices = list(range(start, end))

    # Now create the sliced batch
    device = torch.device("cuda", device_idx)
    sliced_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            sliced_batch[k] = v[sliced_indices].to(device)
        elif isinstance(v, list):
            # If the list's length matches the batch size, slice it.
            if len(v) == B:
                sliced_batch[k] = [v[i] for i in sliced_indices]
            else:
                # Otherwise, leave it unchanged.
                sliced_batch[k] = v
        else:
            # For other types, leave them unchanged.
            sliced_batch[k] = v

    return sliced_batch


# def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
#     if tensor.size(dim) >= length:
#         return tensor
#     else:
#         pad_size = list(tensor.shape)
#         pad_size[dim] = length - tensor.size(dim)
#         return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: int, dim: int = -1
) -> torch.Tensor:
    """Pad tensor to specified length along specified dimension"""
    if tensor.size(dim) >= length:
        return tensor

    pad_size = list(tensor.size())
    pad_size[dim] = length - tensor.size(dim)

    return torch.cat(
        [
            tensor,
            torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device),
        ],
        dim=dim,
    )


def all_gather_if_needed(
    values: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ""):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print("*" * 40)
            print(
                f"[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB"
            )
        print("*" * 40)


def get_block_class_from_model(
    model: torch.nn.Module, block_class_name: str
) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(
    model_class: Type, block_class_name: str
) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith(".py"), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find("transformers") :].replace("/", ".")[:-3]
    print(
        f"Searching in file {filepath}, module {module_name} for class {block_class_name}"
    )

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    port: int = 12355,
    backend: str = "nccl",
):
    print(rank, "initializing distributed")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)
