#!/usr/bin/env python3
"""Evaluate zero-shot MATH performance with the r1_zero prompt."""
"""
uv run python scripts/evaluate_math_baseline.py --model-name-or-path model/Qwen2.5-Math-1.5B --validation-path data/MATH/validation.jsonl
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Local path or HF name for the model to evaluate.",
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=Path("/data/a5-alignment/MATH/validation.jsonl"),
        help="Path to the assignment-style validation JSONL.",
    )
    parser.add_argument(
        "--hendrycks-math-root",
        type=Path,
        default=None,
        help=(
            "Fallback root containing subject folders like "
            "`algebra/test-00000-of-00001.parquet`."
        ),
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("cs336_alignment/prompts/r1_zero.prompt"),
        help="Path to the r1_zero prompt template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write serialized generations and summary metrics.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use with vLLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens per example.",
    )
    parser.add_argument(
        "--stop-string",
        type=str,
        default="</answer>",
        help="Stop generation when this string is emitted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3333333333333333,
        help="Optional cap on the number of validation examples.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def load_hendrycks_math_test_split(root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to load fallback Hendrycks MATH parquet files."
        ) from exc

    parquet_paths = sorted(root.glob("*/test-00000-of-00001.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No Hendrycks MATH test parquet files found under {root}"
        )

    examples: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        subject = parquet_path.parent.name
        for row in table.to_pylist():
            examples.append(
                {
                    **row,
                    "subject": subject,
                    "source_path": str(parquet_path),
                }
            )
    return examples


def load_validation_examples(
    validation_path: Path,
    hendrycks_math_root: Path | None,
) -> list[dict[str, Any]]:
    if validation_path.exists():
        logger.info("Loading validation examples from %s", validation_path)
        return load_jsonl(validation_path)
    if hendrycks_math_root is not None:
        logger.info(
            "Validation path %s is missing; falling back to Hendrycks MATH parquet root %s",
            validation_path,
            hendrycks_math_root,
        )
        return load_hendrycks_math_test_split(hendrycks_math_root)
    raise FileNotFoundError(
        f"Validation path {validation_path} does not exist and no fallback root was provided."
    )


def extract_question(example: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"Could not find a question field in example keys: {sorted(example)}")


def extract_ground_truth(example: dict[str, Any]) -> str | list[str] | int | float:
    for key in ("solution", "ground_truth", "answer", "target", "response"):
        if key in example:
            return example[key]
    raise KeyError(
        f"Could not find a ground-truth field in example keys: {sorted(example)}"
    )


def load_prompt_template(path: Path) -> str:
    return path.read_text()


def format_prompts(prompt_template: str, examples: list[dict[str, Any]]) -> list[str]:
    return [prompt_template.format(question=extract_question(example)) for example in examples]


def categorize_metrics(metrics: dict[str, float]) -> str:
    format_reward = float(metrics["format_reward"])
    answer_reward = float(metrics["answer_reward"])
    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct_format_and_answer"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "correct_format_only"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "incorrect_format"
    raise ValueError(f"Unexpected metric combination: {metrics}")


def summarize_metrics(
    all_metrics: list[dict[str, float]],
    category_counts: Counter[str],
) -> dict[str, Any]:
    summary = {
        "num_examples": len(all_metrics),
        "format_reward_mean": mean(metric["format_reward"] for metric in all_metrics),
        "answer_reward_mean": mean(metric["answer_reward"] for metric in all_metrics),
        "reward_mean": mean(metric["reward"] for metric in all_metrics),
        "category_counts": dict(category_counts),
    }
    return summary


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str | list[str] | int | float],
    eval_sampling_params: SamplingParams,
    examples: list[dict[str, Any]],
    output_path: Path,
    model_name_or_path: str,
) -> dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_responses = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=True)

    all_metrics: list[dict[str, float]] = []
    category_counts: Counter[str] = Counter()
    with output_path.open("w") as fout:
        for idx, (example, prompt, ground_truth, raw_response) in enumerate(
            zip(examples, prompts, ground_truths, raw_responses)
        ):
            generation = raw_response.outputs[0]
            # logger.info("generation: %s", generation)
            response_text = generation.text
            # logger.info("response_text: %s", response_text)
            metrics = reward_fn(response_text, ground_truth)
            # logger.info("metrics: %s", metrics)
            category = categorize_metrics(metrics)
            category_counts[category] += 1
            all_metrics.append(metrics)

            serialized = {
                "example_index": idx,
                "model_name_or_path": model_name_or_path,
                "prompt": prompt,
                "output": response_text,
                "finish_reason": getattr(generation, "finish_reason", None),
                "stop_reason": getattr(generation, "stop_reason", None),
                "num_output_tokens": len(getattr(generation, "token_ids", [])),
                "metrics": metrics,
                "category": category,
                **example,
            }
            fout.write(json.dumps(serialized, ensure_ascii=False) + "\n")

    return summarize_metrics(all_metrics, category_counts)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    prompt_template = load_prompt_template(args.prompt_path)
    examples = load_validation_examples(args.validation_path, args.hendrycks_math_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    logger.info("Loaded %d validation examples", len(examples))

    prompts = format_prompts(prompt_template, examples)
    # if prompts:
        # logger.info("First prompt:\n%s", prompts[0])
    ground_truths = [extract_ground_truth(example) for example in examples]
    # if ground_truths:
        # logger.info("First ground_truths:\n%s", ground_truths[0])
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[args.stop_string],
        include_stop_str_in_output=True,
    )

    logger.info("Loading vLLM model from %s", args.model_name_or_path)
    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"

    summary = evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        examples=examples,
        output_path=results_path,
        model_name_or_path=args.model_name_or_path,
    )
    summary["validation_path"] = str(args.validation_path)
    summary["hendrycks_math_root"] = (
        str(args.hendrycks_math_root) if args.hendrycks_math_root is not None else None
    )
    summary["prompt_path"] = str(args.prompt_path)
    summary["results_path"] = str(results_path)
    summary["sampling"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stop": args.stop_string,
        "include_stop_str_in_output": True,
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote results to %s", results_path)
    logger.info("Wrote summary to %s", summary_path)
    for key, value in summary.items():
        if isinstance(value, (int, float, str)):
            logger.info("%s: %s", key, value)
    logger.info("category_counts: %s", summary["category_counts"])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))
    main()
    logger.info("finished running %s", sys.argv[0])
