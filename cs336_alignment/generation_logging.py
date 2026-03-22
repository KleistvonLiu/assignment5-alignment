from __future__ import annotations

import copy
import json
import logging
from statistics import mean
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output

logger = logging.getLogger(__name__)


def _prepare_tokenizer_for_generation(
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token_id is not None:
        return tokenizer
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token for generation")
    tokenizer = copy.copy(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _maybe_generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    generation_kwargs: dict[str, Any] | None,
) -> list[str]:
    generation_kwargs = dict(generation_kwargs or {})
    generation_kwargs.setdefault("max_new_tokens", 256)
    generation_kwargs.setdefault("do_sample", False)
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        generation_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    device = next(model.parameters()).device
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    prompt_width = input_ids.shape[1]
    response_ids = generated_ids[:, prompt_width:]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)


def _to_display_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(mean(values))


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[Any],
    reward_fn: Callable[[str, Any], dict[str, float]],
    responses: list[str] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    logger_: logging.Logger | None = None,
    log_prefix: str = "eval",
    step: int | None = None,
    num_examples_to_log: int | None = None,
    log_example_details: bool = True,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Generate or score responses and log per-example diagnostics.

    Logged per example:
    - prompt
    - generated response
    - ground truth
    - reward dictionary
    - average token entropy over the response
    - response length in tokens

    Summary stats are returned and optionally sent to `wandb_run`.
    Response length is measured in tokens, and "correct" means `reward == 1.0`.
    """
    if len(prompts) != len(ground_truths):
        raise ValueError("prompts and ground_truths must have the same length")
    if num_examples_to_log is not None:
        prompts = prompts[:num_examples_to_log]
        ground_truths = ground_truths[:num_examples_to_log]

    if not prompts:
        raise ValueError("prompts must be non-empty")

    tokenizer = _prepare_tokenizer_for_generation(tokenizer)
    active_logger = logger_ or logger

    was_training = model.training
    model.eval()
    try:
        if responses is None:
            responses = _maybe_generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                generation_kwargs=generation_kwargs,
            )
        if len(responses) != len(prompts):
            raise ValueError("responses and prompts must have the same length")

        tokenized = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer,
        )
        with torch.inference_mode():
            score_outputs = get_response_log_probs(
                model=model,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
                return_token_entropy=True,
            )

        response_mask = tokenized["response_mask"]
        token_entropy = score_outputs["token_entropy"].detach().cpu()
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.float32)
        avg_token_entropy = (
            (token_entropy * response_mask.to(dtype=token_entropy.dtype)).sum(dim=-1)
            / response_lengths.clamp_min(1.0)
        )

        records: list[dict[str, Any]] = []
        correct_lengths: list[float] = []
        incorrect_lengths: list[float] = []
        reward_values: list[float] = []
        format_reward_values: list[float] = []
        answer_reward_values: list[float] = []

        for idx, (prompt, response, ground_truth, entropy_value, response_length) in enumerate(
            zip(prompts, responses, ground_truths, avg_token_entropy.tolist(), response_lengths.tolist())
        ):
            metrics = reward_fn(response, ground_truth)
            reward_value = float(metrics["reward"])
            format_reward = float(metrics.get("format_reward", float("nan")))
            answer_reward = float(metrics.get("answer_reward", float("nan")))

            reward_values.append(reward_value)
            format_reward_values.append(format_reward)
            answer_reward_values.append(answer_reward)
            if reward_value == 1.0:
                correct_lengths.append(response_length)
            else:
                incorrect_lengths.append(response_length)

            record = {
                "index": idx,
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "metrics": metrics,
                "avg_token_entropy": float(entropy_value),
                "response_length": float(response_length),
            }
            records.append(record)

            if log_example_details:
                active_logger.info(
                    "%s generation %d\nprompt: %s\nresponse: %s\nground_truth: %s\nmetrics: %s\navg_token_entropy: %.4f\nresponse_length: %.1f",
                    log_prefix,
                    idx,
                    prompt,
                    response,
                    _to_display_text(ground_truth),
                    metrics,
                    entropy_value,
                    response_length,
                )
            else:
                active_logger.info(
                    "%s generation %d | metrics=%s | avg_token_entropy=%.4f | response_length=%.1f",
                    log_prefix,
                    idx,
                    metrics,
                    entropy_value,
                    response_length,
                )

        summary = {
            "num_examples": len(records),
            "avg_reward": _mean_or_nan(reward_values),
            "avg_format_reward": _mean_or_nan(format_reward_values),
            "avg_answer_reward": _mean_or_nan(answer_reward_values),
            "avg_token_entropy": _mean_or_nan([record["avg_token_entropy"] for record in records]),
            "avg_response_length": _mean_or_nan([record["response_length"] for record in records]),
            "avg_response_length_correct": _mean_or_nan(correct_lengths),
            "avg_response_length_incorrect": _mean_or_nan(incorrect_lengths),
        }

        active_logger.info("%s generation summary: %s", log_prefix, summary)

        if wandb_run is not None:
            import wandb

            table = wandb.Table(
                columns=[
                    "index",
                    "prompt",
                    "response",
                    "ground_truth",
                    "reward",
                    "format_reward",
                    "answer_reward",
                    "avg_token_entropy",
                    "response_length",
                ]
            )
            for record in records:
                metrics = record["metrics"]
                table.add_data(
                    record["index"],
                    record["prompt"],
                    record["response"],
                    _to_display_text(record["ground_truth"]),
                    float(metrics.get("reward", float("nan"))),
                    float(metrics.get("format_reward", float("nan"))),
                    float(metrics.get("answer_reward", float("nan"))),
                    record["avg_token_entropy"],
                    record["response_length"],
                )

            wandb_payload = {f"{log_prefix}/{key}": value for key, value in summary.items()}
            wandb_payload[f"{log_prefix}/generations"] = table
            if step is not None:
                wandb_payload["step"] = step
            wandb_run.log(wandb_payload)

        return {"records": records, "summary": summary}
    finally:
        if was_training:
            model.train()
