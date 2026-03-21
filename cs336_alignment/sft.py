from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute next-token entropies over the vocabulary dimension."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Masked sum divided by a fixed normalization constant."""
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    return masked_tensor.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run one SFT microbatch loss computation and backward pass."""
    per_example_log_prob = masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )
    loss = -per_example_log_prob.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), {"per_example_log_prob": per_example_log_prob.detach()}


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score each label token under the model's next-token distribution."""
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    labels = labels.to(model_device)

    logits = model(input_ids=input_ids).logits
    full_log_probs = F.log_softmax(logits, dim=-1)
    gathered_log_probs = full_log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    outputs = {"log_probs": gathered_log_probs}
    if return_token_entropy:
        outputs["token_entropy"] = compute_entropy(logits)
    return outputs


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompts and outputs separately, then concatenate them.

    Returns tensors suitable for causal language modeling, where `labels` are
    the one-token-right-shifted concatenated ids and `response_mask` is `True`
    exactly on label positions corresponding to response tokens.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id

    prompt_token_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    output_token_ids = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]

    prompt_and_output_lens = [
        len(prompt_ids) + len(output_ids)
        for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids)
    ]
    max_prompt_and_output_len = max(prompt_and_output_lens)

    batch_size = len(prompt_strs)
    prompt_and_output_ids = torch.full(
        (batch_size, max_prompt_and_output_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    response_mask_full = torch.zeros(
        (batch_size, max_prompt_and_output_len),
        dtype=torch.bool,
    )

    for row_idx, (prompt_ids, output_ids) in enumerate(zip(prompt_token_ids, output_token_ids)):
        prompt_len = len(prompt_ids)
        output_len = len(output_ids)
        concatenated_ids = prompt_ids + output_ids

        prompt_and_output_ids[row_idx, : len(concatenated_ids)] = torch.tensor(
            concatenated_ids,
            dtype=torch.long,
        )
        response_mask_full[row_idx, prompt_len : prompt_len + output_len] = True

    return {
        "input_ids": prompt_and_output_ids[:, :-1],
        "labels": prompt_and_output_ids[:, 1:],
        "response_mask": response_mask_full[:, 1:],
    }
