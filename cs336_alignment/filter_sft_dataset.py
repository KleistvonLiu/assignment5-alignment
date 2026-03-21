from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_sft_examples(
    examples: list[dict[str, Any]],
    reward_fn: Callable[[str, Any], dict[str, float]],
    criterion: str = "reward",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if criterion not in {"reward", "answer_reward"}:
        raise ValueError("criterion must be one of {'reward', 'answer_reward'}")

    kept_examples: list[dict[str, Any]] = []
    kept_metric_values: list[float] = []
    format_reward_sum = 0.0
    answer_reward_sum = 0.0
    reward_sum = 0.0

    for example in examples:
        if "response" not in example:
            raise KeyError("Each example must contain a `response` field")
        if "ground_truth" not in example:
            raise KeyError("Each example must contain a `ground_truth` field for filtering")

        metrics = reward_fn(example["response"], example["ground_truth"])
        format_reward_sum += float(metrics["format_reward"])
        answer_reward_sum += float(metrics["answer_reward"])
        reward_sum += float(metrics["reward"])

        if float(metrics[criterion]) == 1.0:
            kept_examples.append(example)
            kept_metric_values.append(float(metrics[criterion]))

    total_examples = len(examples)
    summary = {
        "criterion": criterion,
        "num_examples_before": total_examples,
        "num_examples_after": len(kept_examples),
        "retention_rate": (len(kept_examples) / total_examples) if total_examples else 0.0,
        "format_reward_mean_before": (format_reward_sum / total_examples) if total_examples else 0.0,
        "answer_reward_mean_before": (answer_reward_sum / total_examples) if total_examples else 0.0,
        "reward_mean_before": (reward_sum / total_examples) if total_examples else 0.0,
    }
    return kept_examples, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input JSONL with at least prompt/response/ground_truth fields.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output JSONL containing only retained examples.",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="reward",
        choices=["reward", "answer_reward"],
        help="Filtering criterion. `reward` is strict; `answer_reward` ignores format mismatches.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path for a JSON summary. Defaults to <output>.summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.input_path)
    kept_examples, summary = filter_sft_examples(
        examples=examples,
        reward_fn=r1_zero_reward_fn,
        criterion=args.criterion,
    )

    summary_path = args.summary_path
    if summary_path is None:
        summary_path = args.output_path.with_suffix(args.output_path.suffix + ".summary.json")

    write_jsonl(args.output_path, kept_examples)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    logger.info("Wrote %d/%d filtered SFT examples to %s", len(kept_examples), len(examples), args.output_path)
    logger.info("Wrote filter summary to %s", summary_path)
    logger.info("Filter summary: %s", summary)


def cli() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))
    main()
    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    cli()
