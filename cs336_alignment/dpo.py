from __future__ import annotations

from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from cs336_alignment.alignment_prompts import render_alpaca_prompt
from cs336_alignment.sft import get_response_log_probs


def _score_response(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
) -> Tensor:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id for DPO scoring")

    full_text = render_alpaca_prompt(instruction=instruction, response=response).rstrip()
    token_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"] + [eos_token_id]
    input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)
    labels = torch.tensor([token_ids[1:]], dtype=torch.long)
    log_probs = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=False,
    )["log_probs"]
    return log_probs.sum(dim=-1).squeeze(0)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    policy_device = next(lm.parameters()).device

    chosen_log_prob = _score_response(
        model=lm,
        tokenizer=tokenizer,
        instruction=prompt,
        response=response_chosen,
    )
    rejected_log_prob = _score_response(
        model=lm,
        tokenizer=tokenizer,
        instruction=prompt,
        response=response_rejected,
    )

    with torch.no_grad():
        ref_chosen_log_prob = _score_response(
            model=lm_ref,
            tokenizer=tokenizer,
            instruction=prompt,
            response=response_chosen,
        )
        ref_rejected_log_prob = _score_response(
            model=lm_ref,
            tokenizer=tokenizer,
            instruction=prompt,
            response=response_rejected,
        )

    ref_chosen_log_prob = ref_chosen_log_prob.to(policy_device)
    ref_rejected_log_prob = ref_rejected_log_prob.to(policy_device)

    preference_logit = (chosen_log_prob - rejected_log_prob) - (
        ref_chosen_log_prob - ref_rejected_log_prob
    )
    return -F.logsigmoid(beta * preference_logit)


def compute_preference_classification_accuracy(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[dict[str, Any]],
) -> float:
    if not examples:
        return 0.0

    correct = 0
    with torch.no_grad():
        for example in examples:
            chosen_score = _score_response(
                model=lm,
                tokenizer=tokenizer,
                instruction=example["instruction"],
                response=example["response_chosen"],
            )
            rejected_score = _score_response(
                model=lm,
                tokenizer=tokenizer,
                instruction=example["instruction"],
                response=example["response_rejected"],
            )
            ref_chosen_score = _score_response(
                model=lm_ref,
                tokenizer=tokenizer,
                instruction=example["instruction"],
                response=example["response_chosen"],
            ).to(chosen_score.device)
            ref_rejected_score = _score_response(
                model=lm_ref,
                tokenizer=tokenizer,
                instruction=example["instruction"],
                response=example["response_rejected"],
            ).to(chosen_score.device)
            policy_margin = (chosen_score - rejected_score) - (
                ref_chosen_score - ref_rejected_score
            )
            if float(policy_margin.item()) > 0.0:
                correct += 1
    return correct / len(examples)


def summarize_losses(losses: list[float]) -> dict[str, float]:
    if not losses:
        return {"count": 0.0, "loss_mean": 0.0}
    return {"count": float(len(losses)), "loss_mean": mean(losses)}
