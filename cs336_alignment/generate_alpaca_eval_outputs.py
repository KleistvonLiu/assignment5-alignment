from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from cs336_alignment.alignment_data import load_alpaca_eval_examples
from cs336_alignment.alignment_eval_utils import (
    generate_outputs,
    get_default_stop_strings,
    init_vllm_model,
    make_sampling_params,
    write_json,
)
from cs336_alignment.alignment_prompts import PromptStyle, render_prompt


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/alpaca_eval/alpaca_eval.jsonl"))
    parser.add_argument(
        "--prompt-style",
        choices=[style.value for style in PromptStyle],
        default=PromptStyle.QUESTION_ONLY.value,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--generator-name", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--stop-string", action="append", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def generate_alpaca_eval_outputs(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_alpaca_eval_examples(args.data_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    prompt_style = PromptStyle(args.prompt_style)
    prompts = [render_prompt(prompt_style, example["instruction"]) for example in examples]

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

    generator_name = args.generator_name or Path(args.model_name_or_path).name
    serialized_outputs: list[dict[str, Any]] = []
    for example, prompt, output in zip(examples, prompts, outputs):
        serialized_outputs.append(
            {
                "instruction": example["instruction"],
                "output": output["text"],
                "generator": generator_name,
                "dataset": example["dataset"],
                "example_id": example["example_id"],
                "prompt": prompt,
                "num_output_tokens": output["num_output_tokens"],
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(serialized_outputs, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "benchmark": "alpaca_eval",
        "model_name_or_path": args.model_name_or_path,
        "generator_name": generator_name,
        "prompt_style": prompt_style.value,
        "output_path": str(args.output_path),
        **generation_summary,
    }
    write_json(args.summary_path or args.output_path.with_name("summary.json"), summary)
    return summary


def main() -> None:
    summary = generate_alpaca_eval_outputs(parse_args())
    logger.info("AlpacaEval generation summary: %s", json.dumps(summary, indent=2))


def cli() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()


if __name__ == "__main__":
    cli()
