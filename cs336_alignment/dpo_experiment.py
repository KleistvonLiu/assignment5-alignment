from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import RMSprop
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.alignment_data import DEFAULT_HH_PATH, load_hh_preferences, split_train_validation
from cs336_alignment.dpo import (
    compute_per_instance_dpo_loss,
    compute_preference_classification_accuracy,
)


logger = logging.getLogger(__name__)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--hh-path", type=Path, default=DEFAULT_HH_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--policy-device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reference-device",
        type=str,
        default="cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_model_and_tokenizer(model_name_or_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if device.startswith("cuda") and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def evaluate_validation_metrics(
    policy_model,
    reference_model,
    tokenizer,
    validation_examples: list[dict[str, Any]],
    beta: float,
) -> dict[str, float]:
    if not validation_examples:
        return {"eval/val_loss": 0.0, "eval/val_accuracy": 0.0}

    policy_model.eval()
    reference_model.eval()
    losses: list[float] = []
    for example in validation_examples:
        loss = compute_per_instance_dpo_loss(
            lm=policy_model,
            lm_ref=reference_model,
            tokenizer=tokenizer,
            beta=beta,
            prompt=example["instruction"],
            response_chosen=example["response_chosen"],
            response_rejected=example["response_rejected"],
        )
        losses.append(float(loss.item()))
    accuracy = compute_preference_classification_accuracy(
        lm=policy_model,
        lm_ref=reference_model,
        tokenizer=tokenizer,
        examples=validation_examples,
    )
    policy_model.train()
    return {
        "eval/val_loss": sum(losses) / max(len(losses), 1),
        "eval/val_accuracy": accuracy,
    }


def save_checkpoint(model, tokenizer, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def train_dpo(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.train_batch_size != args.gradient_accumulation_steps:
        logger.warning(
            "This implementation trains one example at a time; effective batch size equals gradient_accumulation_steps."
        )

    policy_model, tokenizer = _load_model_and_tokenizer(args.model_name_or_path, args.policy_device)
    reference_model, _ = _load_model_and_tokenizer(args.model_name_or_path, args.reference_device)
    policy_model.train()
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    all_examples = load_hh_preferences(args.hh_path)
    if args.max_train_examples is not None:
        all_examples = all_examples[: args.max_train_examples]
    train_examples, validation_examples = split_train_validation(
        examples=all_examples,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    optimizer = RMSprop(policy_model.parameters(), lr=args.learning_rate)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    best_val_accuracy = float("-inf")
    for epoch in range(args.num_train_epochs):
        random.shuffle(train_examples)
        progress = tqdm(train_examples, desc=f"dpo epoch {epoch + 1}/{args.num_train_epochs}")
        running_losses: list[float] = []
        for example_idx, example in enumerate(progress, start=1):
            loss = compute_per_instance_dpo_loss(
                lm=policy_model,
                lm_ref=reference_model,
                tokenizer=tokenizer,
                beta=args.beta,
                prompt=example["instruction"],
                response_chosen=example["response_chosen"],
                response_rejected=example["response_rejected"],
            )
            (loss / args.gradient_accumulation_steps).backward()
            running_losses.append(float(loss.item()))

            if example_idx % args.gradient_accumulation_steps != 0 and example_idx != len(train_examples):
                continue

            grad_norm = float(clip_grad_norm_(policy_model.parameters(), args.max_grad_norm).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            train_metrics = {
                "train_step": global_step,
                "epoch": epoch + 1,
                "train/loss": sum(running_losses) / max(len(running_losses), 1),
                "train/grad_norm": grad_norm,
            }
            append_jsonl(train_metrics_path, train_metrics)
            running_losses.clear()

            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                eval_metrics = evaluate_validation_metrics(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    tokenizer=tokenizer,
                    validation_examples=validation_examples,
                    beta=args.beta,
                )
                payload = {"eval_step": global_step, "epoch": epoch + 1, **eval_metrics}
                append_jsonl(eval_metrics_path, payload)
                if eval_metrics["eval/val_accuracy"] > best_val_accuracy:
                    best_val_accuracy = eval_metrics["eval/val_accuracy"]
                    save_checkpoint(policy_model, tokenizer, output_dir / "best_model")

    final_metrics = evaluate_validation_metrics(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        validation_examples=validation_examples,
        beta=args.beta,
    )
    save_checkpoint(policy_model, tokenizer, output_dir / "final_model")
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "hh_path": str(args.hh_path),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "final_val_loss": final_metrics["eval/val_loss"],
        "final_val_accuracy": final_metrics["eval/val_accuracy"],
        "best_val_accuracy": best_val_accuracy if best_val_accuracy != float("-inf") else final_metrics["eval/val_accuracy"],
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    summary = train_dpo(parse_args())
    logger.info("DPO summary: %s", json.dumps(summary, indent=2))


def cli() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()


if __name__ == "__main__":
    cli()
