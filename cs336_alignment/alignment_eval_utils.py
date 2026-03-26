from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from cs336_alignment.alignment_metrics import summarize_length_stats
from cs336_alignment.alignment_prompts import PromptStyle


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate)
    return model_name_or_path


def get_default_stop_strings(prompt_style: PromptStyle | str) -> list[str] | None:
    style = PromptStyle(prompt_style)
    if style == PromptStyle.ZERO_SHOT:
        return ["# Query:"]
    return None


def make_sampling_params(
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop_strings: list[str] | None,
):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop_strings:
        kwargs["stop"] = stop_strings
    return SamplingParams(**kwargs)


def init_vllm_model(
    *,
    model_name_or_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
):
    from vllm import LLM

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return LLM(
        model=resolve_model_name_or_path(model_name_or_path),
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def generate_outputs(
    *,
    llm,
    prompts: list[str],
    sampling_params,
    use_tqdm: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    start_time = time.perf_counter()
    raw_outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    generation_seconds = time.perf_counter() - start_time

    outputs: list[dict[str, Any]] = []
    output_lengths: list[int] = []
    for raw_output in raw_outputs:
        generated = raw_output.outputs[0]
        token_count = len(getattr(generated, "token_ids", []))
        output_lengths.append(token_count)
        outputs.append(
            {
                "text": generated.text,
                "num_output_tokens": token_count,
                "finish_reason": getattr(generated, "finish_reason", None),
                "stop_reason": getattr(generated, "stop_reason", None),
            }
        )

    summary = {
        "generation_seconds": generation_seconds,
        "num_examples": float(len(prompts)),
        "examples_per_second": (len(prompts) / generation_seconds) if generation_seconds > 0 else 0.0,
        **summarize_length_stats(output_lengths),
    }
    return outputs, summary


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_boolean_metric(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not records:
        return {f"{key}_mean": 0.0}
    return {f"{key}_mean": mean(float(record[key]) for record in records)}
