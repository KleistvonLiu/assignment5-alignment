from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from unittest.mock import patch

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_math_baseline import (
    evaluate_vllm,
    extract_ground_truth,
    format_prompts,
    load_prompt_template,
    load_validation_examples,
)
from cs336_alignment.generation_logging import log_generations
from cs336_alignment.sft import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output

if TYPE_CHECKING:
    from vllm import LLM
    from vllm import SamplingParams

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        help="Base model path or HF repo id.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("/data/a5-alignment/MATH/sft.jsonl"),
        help="Path to the reasoning SFT dataset.",
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=Path("/data/a5-alignment/MATH/validation.jsonl"),
        help="Path to the MATH validation JSONL.",
    )
    parser.add_argument(
        "--hendrycks-math-root",
        type=Path,
        default=None,
        help="Optional fallback parquet root for validation examples.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("cs336_alignment/prompts/r1_zero.prompt"),
        help="Validation prompt template used during periodic eval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, logs, and evaluation outputs.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Optional cap on the number of SFT examples.",
    )
    parser.add_argument(
        "--filter-correct-only",
        action="store_true",
        help="Keep only examples whose response receives reward 1. Requires a ground-truth key.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy-device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--vllm-device",
        type=str,
        default="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",
    )
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--num-eval-examples", type=int, default=256)
    parser.add_argument("--num-log-generations", type=int, default=4)
    parser.add_argument("--save-every-eval", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def resolve_path_with_local_fallback(path: Path, local_fallback: Path | None = None) -> Path:
    if path.exists():
        return path
    if local_fallback is not None and local_fallback.exists():
        return local_fallback
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def maybe_filter_correct_examples(
    examples: list[dict[str, Any]],
    reward_fn: Callable[[str, Any], dict[str, float]],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for example in examples:
        if "response" not in example:
            raise KeyError("Expected each SFT example to contain a `response` field")
        ground_truth = extract_ground_truth(example)
        metrics = reward_fn(example["response"], ground_truth)
        if float(metrics["reward"]) == 1.0:
            filtered.append(example)
    return filtered


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_prompt_response_batch(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    prompts = [example["prompt"] for example in examples]
    responses = [example["response"] for example in examples]
    tokenized = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    tokenized["examples"] = examples
    tokenized["prompts"] = prompts
    tokenized["responses"] = responses
    return tokenized


def build_sft_dataloader(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_prompt_response_batch(batch, tokenizer),
    )


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def setup_wandb(args: argparse.Namespace) -> Any | None:
    if args.wandb_project is None:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return run


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    checkpoint_name: str,
) -> Path:
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def load_logged_responses(results_path: Path, limit: int) -> list[str]:
    responses: list[str] = []
    with results_path.open() as f:
        for line in f:
            if len(responses) >= limit:
                break
            responses.append(json.loads(line)["output"])
    return responses


def evaluate_policy(
    *,
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    llm: LLM,
    validation_examples: list[dict[str, Any]],
    prompt_template: str,
    model_name_or_path: str,
    output_dir: Path,
    eval_step: int,
    num_eval_examples: int,
    num_log_generations: int,
    wandb_run: Any | None,
) -> dict[str, Any]:
    from vllm import SamplingParams

    eval_examples = validation_examples[:num_eval_examples]
    prompts = format_prompts(prompt_template, eval_examples)
    ground_truths = [extract_ground_truth(example) for example in eval_examples]

    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    eval_dir = output_dir / f"eval_step_{eval_step:06d}"
    results_path = eval_dir / "results.jsonl"
    summary_path = eval_dir / "summary.json"
    summary = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        examples=eval_examples,
        output_path=results_path,
        model_name_or_path=model_name_or_path,
    )
    summary["eval_step"] = eval_step
    summary["results_path"] = str(results_path)
    summary["num_eval_examples"] = len(eval_examples)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    generation_log_summary = None
    if num_log_generations > 0:
        responses = load_logged_responses(results_path, limit=num_log_generations)
        generation_log_output = log_generations(
            model=policy,
            tokenizer=tokenizer,
            prompts=prompts[: len(responses)],
            ground_truths=ground_truths[: len(responses)],
            responses=responses,
            reward_fn=r1_zero_reward_fn,
            log_prefix="eval",
            step=eval_step,
            wandb_run=wandb_run,
        )
        generation_log_summary = generation_log_output["summary"]
        summary["logged_generation_summary"] = generation_log_summary

    if wandb_run is not None:
        wandb_payload = {
            "eval_step": eval_step,
            "eval/answer_reward_mean": summary["answer_reward_mean"],
            "eval/format_reward_mean": summary["format_reward_mean"],
            "eval/reward_mean": summary["reward_mean"],
        }
        if generation_log_summary is not None:
            wandb_payload["eval/avg_token_entropy"] = generation_log_summary["avg_token_entropy"]
            wandb_payload["eval/avg_response_length"] = generation_log_summary["avg_response_length"]
        for category, count in summary["category_counts"].items():
            wandb_payload[f"eval/{category}"] = count
        wandb_run.log(wandb_payload)

    logger.info("eval step %d summary: %s", eval_step, summary)
    return summary


def train_sft(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_path = resolve_path_with_local_fallback(args.train_path)
    validation_path = resolve_path_with_local_fallback(
        args.validation_path,
        local_fallback=Path("data/MATH/validation.jsonl"),
    )
    model_name_or_path = args.model_name_or_path
    if not Path(model_name_or_path).exists():
        local_model_path = Path("model/Qwen2.5-Math-1.5B")
        if local_model_path.exists():
            model_name_or_path = str(local_model_path)

    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")
    micro_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"

    train_examples = load_jsonl(train_path)
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    if args.filter_correct_only:
        before = len(train_examples)
        train_examples = maybe_filter_correct_examples(train_examples, r1_zero_reward_fn)
        logger.info("Filtered SFT examples from %d to %d", before, len(train_examples))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if "cuda" in args.policy_device and torch.cuda.is_available() else torch.float32
    attn_implementation = "flash_attention_2" if "cuda" in args.policy_device and torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model.to(args.policy_device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    prompt_template = load_prompt_template(args.prompt_path)
    validation_examples = load_validation_examples(validation_path, args.hendrycks_math_root)
    wandb_run = setup_wandb(args)

    llm = None
    if args.eval_every_steps > 0 and args.num_eval_examples > 0:
        llm = init_vllm(
            model_id=model_name_or_path,
            device=args.vllm_device,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.num_train_epochs):
        dataloader = build_sft_dataloader(
            examples=train_examples,
            tokenizer=tokenizer,
            batch_size=micro_batch_size,
            shuffle=True,
        )
        num_microbatches = len(dataloader)
        epoch_iterator = tqdm(
            enumerate(dataloader),
            total=num_microbatches,
            desc=f"epoch {epoch + 1}/{args.num_train_epochs}",
        )

        train_loss_window: list[float] = []
        train_entropy_window: list[float] = []
        for microbatch_idx, batch in epoch_iterator:
            if microbatch_idx % args.gradient_accumulation_steps == 0:
                current_window_size = min(
                    args.gradient_accumulation_steps,
                    num_microbatches - microbatch_idx,
                )

            input_ids = batch["input_ids"].to(args.policy_device)
            labels = batch["labels"].to(args.policy_device)
            response_mask = batch["response_mask"].to(args.policy_device)

            outputs = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=outputs["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=current_window_size,
                normalize_constant=1.0,
            )
            train_loss_window.append(float(loss.item()))

            response_lengths = response_mask.sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0)
            token_entropy = outputs["token_entropy"]
            avg_entropy = (
                (token_entropy * response_mask.to(dtype=token_entropy.dtype)).sum(dim=-1)
                / response_lengths
            ).mean()
            train_entropy_window.append(float(avg_entropy.item()))

            is_window_end = (
                (microbatch_idx + 1) % args.gradient_accumulation_steps == 0
                or microbatch_idx + 1 == num_microbatches
            )
            if not is_window_end:
                continue

            grad_norm = float(clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            train_metrics = {
                "train_step": global_step,
                "epoch": epoch + 1,
                "train/loss": float(sum(train_loss_window) / max(len(train_loss_window), 1)),
                "train/avg_token_entropy": float(sum(train_entropy_window) / max(len(train_entropy_window), 1)),
                "train/grad_norm": grad_norm,
            }
            append_jsonl(train_metrics_path, train_metrics)
            logger.info("train step %d metrics: %s", global_step, train_metrics)
            if wandb_run is not None:
                wandb_run.log(train_metrics)
            train_loss_window.clear()
            train_entropy_window.clear()

            if llm is not None and global_step % args.eval_every_steps == 0:
                eval_summary = evaluate_policy(
                    policy=model,
                    tokenizer=tokenizer,
                    llm=llm,
                    validation_examples=validation_examples,
                    prompt_template=prompt_template,
                    model_name_or_path=model_name_or_path,
                    output_dir=output_dir / "evals",
                    eval_step=global_step,
                    num_eval_examples=args.num_eval_examples,
                    num_log_generations=args.num_log_generations,
                    wandb_run=wandb_run,
                )
                append_jsonl(eval_metrics_path, eval_summary)
                if args.save_every_eval:
                    save_checkpoint(model, tokenizer, output_dir, f"checkpoint_step_{global_step:06d}")

    final_checkpoint = save_checkpoint(model, tokenizer, output_dir, "final_model")
    logger.info("Saved final model to %s", final_checkpoint)
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    train_sft(args)


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
