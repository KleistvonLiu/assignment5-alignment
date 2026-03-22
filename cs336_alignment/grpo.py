from __future__ import annotations

from typing import Any, Callable, Literal

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, Any], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[Any],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute per-rollout rewards and normalize them within each rollout group.

    Each contiguous block of `group_size` responses is treated as one group,
    corresponding to multiple rollouts for the same prompt/question.

    If `normalize_by_std=True`, this implements the standard group-normalized
    reward:

        A_i = (r_i - mean(group)) / (std(group) + advantage_eps)

    Otherwise, it implements the R1-style variant without std normalization:

        A_i = r_i - mean(group)
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rollout_batch_size = len(rollout_responses)
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout batch size must be divisible by group_size")

    raw_reward_values = [
        float(reward_fn(response, ground_truth)["reward"])
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw_reward_values, dtype=torch.float32)

    grouped_raw_rewards = raw_rewards.view(-1, group_size)
    group_means = grouped_raw_rewards.mean(dim=1, keepdim=True)
    centered_rewards = grouped_raw_rewards - group_means

    if normalize_by_std:
        if group_size == 1:
            group_stds = torch.zeros_like(group_means)
        else:
            group_stds = grouped_raw_rewards.std(dim=1, keepdim=True, unbiased=True)
        normalized_rewards = centered_rewards / (group_stds + advantage_eps)
    else:
        group_stds = grouped_raw_rewards.std(dim=1, keepdim=True, unbiased=True) if group_size > 1 else torch.zeros_like(group_means)
        normalized_rewards = centered_rewards

    metadata = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=False).item()) if raw_rewards.numel() > 1 else 0.0,
        "reward_min": float(raw_rewards.min().item()) if raw_rewards.numel() > 0 else 0.0,
        "reward_max": float(raw_rewards.max().item()) if raw_rewards.numel() > 0 else 0.0,
        "group_mean_mean": float(group_means.mean().item()) if group_means.numel() > 0 else 0.0,
        "group_std_mean": float(group_stds.mean().item()) if group_stds.numel() > 0 else 0.0,
        "num_groups": float(grouped_raw_rewards.shape[0]),
    }
    return normalized_rewards.reshape(-1), raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token REINFORCE-style loss from rewards or advantages.

    The scalar reward/advantage for each rollout is broadcast across the
    sequence dimension:

        loss_{i,t} = -A_i * log pi_theta(o_{i,t} | q_i, o_{i,<t})
    """
    if raw_rewards_or_advantages.ndim != 2 or raw_rewards_or_advantages.shape[1] != 1:
        raise ValueError("raw_rewards_or_advantages must have shape (batch_size, 1)")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length)")
    if raw_rewards_or_advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("batch dimensions must match")
    return -raw_rewards_or_advantages.to(dtype=policy_log_probs.dtype) * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-token GRPO-Clip loss.

    Let ratio = pi_theta / pi_theta_old = exp(logp - old_logp). Then the clipped
    objective is:

        -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    """
    if advantages.ndim != 2 or advantages.shape[1] != 1:
        raise ValueError("advantages must have shape (batch_size, 1)")
    if policy_log_probs.ndim != 2 or old_log_probs.ndim != 2:
        raise ValueError("policy_log_probs and old_log_probs must have shape (batch_size, sequence_length)")
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs and old_log_probs must have the same shape")
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("batch dimensions must match")

    advantages = advantages.to(dtype=policy_log_probs.dtype)
    log_ratio = policy_log_probs - old_log_probs.to(dtype=policy_log_probs.dtype)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    used_clipped = clipped_objective < unclipped_objective
    loss = -torch.minimum(unclipped_objective, clipped_objective)

    metadata = {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "used_clipped": used_clipped,
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None,
    advantages: torch.Tensor | None,
    old_log_probs: torch.Tensor | None,
    cliprange: float | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the requested policy-gradient loss routine."""
    metadata: dict[str, torch.Tensor] = {}

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        return loss, metadata

    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        return loss, metadata

    if loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='grpo_clip'")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for loss_type='grpo_clip'")
        if cliprange is None:
            raise ValueError("cliprange is required for loss_type='grpo_clip'")
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean over masked elements only.

    Elements where `mask == 0` do not contribute to either the numerator or the
    denominator. If a slice contains no unmasked elements, the result for that
    slice is `NaN`, matching standard division semantics.
    """
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape")
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    counts = mask.to(dtype=tensor.dtype).sum(dim=dim)
    sums = masked_tensor.sum(dim=dim)
    return sums / counts


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one GRPO microbatch loss computation and backward pass."""
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(
        tensor=per_token_loss,
        mask=response_mask,
        dim=-1,
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    detached_metadata = {key: value.detach() for key, value in metadata.items()}
    detached_metadata["per_token_loss"] = per_token_loss.detach()
    detached_metadata["per_example_loss"] = per_example_loss.detach()
    return loss.detach(), detached_metadata
