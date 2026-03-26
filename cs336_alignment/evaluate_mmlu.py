from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from cs336_alignment.alignment_data import load_mmlu_examples
from cs336_alignment.alignment_eval_utils import (
    generate_outputs,
    get_default_stop_strings,
    init_vllm_model,
    make_sampling_params,
    write_json,
    write_jsonl,
)
from cs336_alignment.alignment_metrics import parse_mmlu_response
from cs336_alignment.alignment_prompts import PromptStyle, render_prompt


logger = logging.getLogger(__name__)


def build_mmlu_instruction(example: dict[str, Any]) -> str:
    options = example["options"]
    return (
        f'Answer the following multiple choice question about {example["subject"]}. '
        'Respond with a single sentence of the form "The correct answer is _", '
        "filling the blank with the letter corresponding to the correct answer "
        "(i.e., A, B, C or D).\n"
        f'Question: {example["question"]}\n'
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        "Answer:"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/mmlu"))
    parser.add_argument(
        "--prompt-style",
        choices=[style.value for style in PromptStyle if style != PromptStyle.QUESTION_ONLY],
        default=PromptStyle.ZERO_SHOT.value,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--stop-string", action="append", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def evaluate_mmlu(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_mmlu_examples(args.data_root)
    if args.limit is not None:
        examples = examples[: args.limit]

    prompt_style = PromptStyle(args.prompt_style)
    prompts = [render_prompt(prompt_style, build_mmlu_instruction(example)) for example in examples]

    llm = init_vllm_model(
        model_name_or_path=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = make_sampling_params(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_strings=args.stop_string or get_default_stop_strings(prompt_style),
    )
    outputs, generation_summary = generate_outputs(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )

    results: list[dict[str, Any]] = []
    parse_failures = 0
    num_correct = 0
    for example, prompt, output in zip(examples, prompts, outputs):
        parsed_prediction = parse_mmlu_response(example, output["text"])
        parse_failed = parsed_prediction is None
        is_correct = parsed_prediction == example["answer"]
        parse_failures += int(parse_failed)
        num_correct += int(is_correct)
        results.append(
            {
                **example,
                "prompt": prompt,
                "output": output["text"],
                "parsed_prediction": parsed_prediction,
                "gold": example["answer"],
                "is_correct": is_correct,
                "parse_failed": parse_failed,
                "num_output_tokens": output["num_output_tokens"],
                "finish_reason": output["finish_reason"],
                "stop_reason": output["stop_reason"],
            }
        )

    summary = {
        "benchmark": "mmlu",
        "model_name_or_path": args.model_name_or_path,
        "prompt_style": prompt_style.value,
        "num_examples": len(results),
        "num_correct": num_correct,
        "accuracy": (num_correct / len(results)) if results else 0.0,
        "num_parse_failures": parse_failures,
        "parse_failure_rate": (parse_failures / len(results)) if results else 0.0,
        **generation_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "results.jsonl", results)
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = evaluate_mmlu(parse_args())
    logger.info("MMLU summary: %s", json.dumps(summary, indent=2))


def cli() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()


if __name__ == "__main__":
    cli()
