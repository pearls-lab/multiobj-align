import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class MultiHeadCausalLM(nn.Module):
    """
    A wrapper that converts a pretrained causal LM into a multi-head model.
    It obtains hidden states by calling the underlying transformer (instead of the full forward
    that calls the original lm_head) and then applies N different linear heads to produce logits.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_heads: int,
        equal_init: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads

        # Determine the precision to use throughout the model
        self.model_dtype = next(base_model.parameters()).dtype
        print(f"Model initialization dtype: {self.model_dtype}")

        # Retrieve hidden size and vocabulary size
        hidden_size = base_model.config.hidden_size
        vocab_size = base_model.config.vocab_size

        # Create a ModuleList for the heads
        self.heads = nn.ModuleList()

        # KEEP ORIGINAL ARCHITECTURE: Always use original lm_head as first head
        if hasattr(base_model, "lm_head"):
            original_head = base_model.lm_head
            self.heads.append(original_head)  # Keep the direct reference
            original_weights = original_head.weight.data.clone()
        else:
            # Create a new head if lm_head doesn't exist
            original_head = nn.Linear(hidden_size, vocab_size, bias=False)
            if hasattr(base_model, "get_output_embeddings"):
                original_head.weight.data.copy_(
                    base_model.get_output_embeddings().weight.data
                )
            self.heads.append(original_head)
            original_weights = original_head.weight.data.clone()

        # Only apply perturbations if equal_init=True
        if equal_init:
            print("Applying perturbations to create diverse heads")

            # Use small perturbation scale for all heads
            perturbation_scale = 0.001

            # Apply perturbation to the original head (first head)
            if hasattr(base_model, "lm_head"):
                torch.manual_seed(42)
                perturbed_weights = (
                    original_weights
                    + perturbation_scale * torch.randn_like(original_weights)
                )
                original_head.weight.data.copy_(perturbed_weights)
                print(f"Applied perturbation (scale={perturbation_scale}) to head 0")

            # Create additional heads with perturbations
            for i in range(1, num_heads):
                new_head = nn.Linear(hidden_size, vocab_size, bias=False)

                # Use same noise scale with different seeds for each head
                torch.manual_seed(42 + i * 100)
                perturbed_weights = (
                    original_weights
                    + perturbation_scale * torch.randn_like(original_weights)
                )
                new_head.weight.data.copy_(perturbed_weights)
                print(f"Applied perturbation (scale={perturbation_scale}) to head {i}")

                # Ensure consistent dtype
                new_head = new_head.to(self.model_dtype)
                self.heads.append(new_head)
        else:
            print("Creating heads WITHOUT perturbations (keeping original SFT weights)")

            # Create additional heads as exact copies (no perturbations)
            for i in range(1, num_heads):
                new_head = nn.Linear(hidden_size, vocab_size, bias=False)

                # Copy original weights exactly (no noise)
                new_head.weight.data.copy_(original_weights)
                print(f"Created unperturbed head {i} (exact copy of SFT head)")

                # Ensure consistent dtype
                new_head = new_head.to(self.model_dtype)
                self.heads.append(new_head)

    @staticmethod
    def _get_hidden_states(base_model, input_ids, attention_mask, **kwargs):
        """Extract hidden states from the base model."""
        # Get device and dtype information from the model
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype

        # Ensure inputs are on the correct device
        input_ids = input_ids.to(device)
        attention_mask = (
            attention_mask.to(device) if attention_mask is not None else None
        )

        # If the base model has a transformer (as in GPT2LMHeadModel), use it
        if hasattr(base_model, "transformer"):
            outputs = base_model.transformer(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            # For GPT2, the transformer returns an object with attribute "last_hidden_state"
            hidden_states = outputs.last_hidden_state
        # If the base model has a model attribute (as in LlamaForCausalLM), use it
        elif hasattr(base_model, "model"):
            outputs = base_model.model(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            # For LLaMA, the model returns an object with attribute "last_hidden_state"
            hidden_states = outputs.last_hidden_state
        else:
            # Otherwise, fallback to calling the full forward and expecting hidden states
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
            hidden_states = (
                outputs.hidden_states[-1]
                if hasattr(outputs, "hidden_states")
                and outputs.hidden_states is not None
                else outputs.last_hidden_state
            )

        # Ensure hidden states have the same dtype as the model
        return hidden_states.to(dtype)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass that applies each head to the hidden states.
        """
        # Get the current model dtype and device
        current_dtype = next(self.parameters()).dtype
        current_device = next(self.parameters()).device

        # Move inputs to the right device
        input_ids = input_ids.to(current_device)
        attention_mask = (
            attention_mask.to(current_device) if attention_mask is not None else None
        )

        # Get hidden states
        hidden_states = self._get_hidden_states(
            self.base_model, input_ids, attention_mask, **kwargs
        )

        # Convert hidden states to current dtype without warning
        if hidden_states.dtype != current_dtype:
            hidden_states = hidden_states.to(current_dtype)

        # Apply each head
        logits_per_head = []
        for i, head in enumerate(self.heads):
            # Ensure head is using the correct dtype (permanently)
            if head.weight.dtype != current_dtype:
                self.heads[i] = head.to(current_dtype)
            # Apply head to hidden states
            logits = self.heads[i](hidden_states)
            logits_per_head.append(logits)

        # Stack along a new dimension for the heads
        multi_head_logits = torch.stack(logits_per_head, dim=0)
        return {"logits": multi_head_logits}

    def generate_with_head(self, head_idx, *args, **kwargs):
        """
        Generate text using a specific head.

        Args:
            head_idx (int): Index of the head to use for generation
            *args, **kwargs: Arguments to pass to the base model's generate method
        """
        if head_idx < 0 or head_idx >= self.num_heads:
            raise ValueError(
                f"Head index {head_idx} out of range (0 to {self.num_heads-1})"
            )

        original_lm_head = getattr(self.base_model, "lm_head", None)
        current_dtype = next(self.parameters()).dtype
        current_device = next(self.parameters()).device

        # Ensure input tensors are on the right device
        for arg_idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args = list(args)
                args[arg_idx] = arg.to(current_device)
                args = tuple(args)

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(current_device)

        # Make sure the selected head has the correct dtype
        selected_head = self.heads[head_idx].to(current_dtype)

        # Temporarily set the model's lm_head
        self.base_model.lm_head = selected_head

        try:
            generated = self.base_model.generate(*args, **kwargs)
        finally:
            # Always restore the original state
            if original_lm_head is not None:
                self.base_model.lm_head = original_lm_head
            else:
                delattr(self.base_model, "lm_head")

        return generated

    def generate_ensemble(self, *args, head_weights=None, **kwargs):
        """
        Generate text using an ensemble of heads with weighted averaging.

        Args:
            head_weights (List[float], optional): Weights for each head.
                If None, equal weights will be used.
            *args, **kwargs: Arguments to pass to the base model's generate method
        """
        # Default to equal weights if not specified
        if head_weights is None:
            head_weights = [1.0 / self.num_heads] * self.num_heads
        else:
            if len(head_weights) != self.num_heads:
                raise ValueError(
                    f"Expected {self.num_heads} weights, got {len(head_weights)}"
                )
            # Normalize weights to sum to 1
            total = sum(head_weights)
            head_weights = [w / total for w in head_weights]

        original_lm_head = getattr(self.base_model, "lm_head", None)
        current_dtype = next(self.parameters()).dtype
        current_device = next(self.parameters()).device

        # Ensure input tensors are on the right device
        for arg_idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args = list(args)
                args[arg_idx] = arg.to(current_device)
                args = tuple(args)

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(current_device)

        # Create a custom head that ensembles the predictions from all heads
        class EnsembleHead(nn.Module):
            def __init__(self, heads, weights, dtype):
                super().__init__()
                self.heads = heads
                self.weights = weights
                self.dtype = dtype
                # Copy the weight attribute from the first head for compatibility
                self.weight = heads[0].weight

            def forward(self, hidden_states):
                # Apply each head and weight the results
                weighted_logits = None
                for i, (head, weight) in enumerate(zip(self.heads, self.weights)):
                    # Ensure head uses the correct dtype
                    head = head.to(self.dtype)

                    # Apply head to get logits
                    logits = head(hidden_states)

                    # Add weighted contribution
                    if weighted_logits is None:
                        weighted_logits = logits * weight
                    else:
                        weighted_logits += logits * weight

                return weighted_logits

        # Create the ensemble head
        ensemble_head = EnsembleHead(
            heads=[head.to(current_dtype) for head in self.heads],
            weights=head_weights,
            dtype=current_dtype,
        ).to(current_device)

        # Temporarily set the model's lm_head
        self.base_model.lm_head = ensemble_head

        try:
            generated = self.base_model.generate(*args, **kwargs)
        finally:
            # Always restore the original state
            if original_lm_head is not None:
                self.base_model.lm_head = original_lm_head
            else:
                delattr(self.base_model, "lm_head")

        return generated
