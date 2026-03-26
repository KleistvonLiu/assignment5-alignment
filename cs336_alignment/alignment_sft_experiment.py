from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from cs336_alignment.alignment_data import (
    DEFAULT_SFT_TRAIN_PATH,
    DEFAULT_SFT_VALID_PATH,
    PackedSFTDataset,
    iterate_batches,
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
    parser.add_argument("--train-path", type=Path, default=DEFAULT_SFT_TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_SFT_VALID_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _resolve_gradient_accumulation_steps(args: argparse.Namespace) -> int:
    if args.gradient_accumulation_steps is None:
        if args.train_batch_size % args.micro_batch_size != 0:
            raise ValueError("train_batch_size must be divisible by micro_batch_size")
        return args.train_batch_size // args.micro_batch_size
    expected_train_batch_size = args.micro_batch_size * args.gradient_accumulation_steps
    if expected_train_batch_size != args.train_batch_size:
        raise ValueError(
            "train_batch_size must equal micro_batch_size * gradient_accumulation_steps"
        )
    return args.gradient_accumulation_steps


def _load_model_and_tokenizer(model_name_or_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {}
    if use_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    return model, tokenizer


def _compute_batch_loss(model, batch: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids).logits
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
    )


@torch.no_grad()
def evaluate_validation_loss(model, dataloader, device: str) -> float | None:
    if dataloader is None:
        return None
    model.eval()
    losses: list[float] = []
    for batch in dataloader:
        loss = _compute_batch_loss(model, batch, device)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def maybe_setup_wandb(args: argparse.Namespace):
    if args.wandb_project is None:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )


def save_checkpoint(model, tokenizer, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def train_alignment_sft(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    accumulation_steps = _resolve_gradient_accumulation_steps(args)

    model, tokenizer = _load_model_and_tokenizer(args.model_name_or_path, args.device)
    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=args.train_path,
        seq_length=args.seq_length,
        shuffle=False,
    )
    if args.max_train_examples is not None:
        train_dataset.examples = train_dataset.examples[: args.max_train_examples]

    valid_dataloader = None
    if args.valid_path.exists():
        valid_dataset = PackedSFTDataset(
            tokenizer=tokenizer,
            dataset_path=args.valid_path,
            seq_length=args.seq_length,
            shuffle=False,
        )
        if args.max_valid_examples is not None:
            valid_dataset.examples = valid_dataset.examples[: args.max_valid_examples]
        valid_dataloader = iterate_batches(
            dataset=valid_dataset,
            batch_size=args.micro_batch_size,
            shuffle=False,
        )
    else:
        logger.warning("Validation path does not exist; skipping validation: %s", args.valid_path)

    train_dataloader = iterate_batches(
        dataset=train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_optimizer_steps = math.ceil(len(train_dataloader) * args.num_train_epochs / accumulation_steps)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_optimizer_steps, 1),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"
    wandb_run = maybe_setup_wandb(args)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(args.num_train_epochs):
        progress = tqdm(train_dataloader, desc=f"sft epoch {epoch + 1}/{args.num_train_epochs}")
        running_losses: list[float] = []
        for micro_step, batch in enumerate(progress, start=1):
            loss = _compute_batch_loss(model, batch, args.device) / accumulation_steps
            loss.backward()
            running_losses.append(float(loss.item() * accumulation_steps))

            if micro_step % accumulation_steps != 0 and micro_step != len(train_dataloader):
                continue

            grad_norm = float(clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            train_metrics = {
                "train_step": global_step,
                "epoch": epoch + 1,
                "train/loss": sum(running_losses) / max(len(running_losses), 1),
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm,
            }
            append_jsonl(train_metrics_path, train_metrics)
            if wandb_run is not None:
                wandb_run.log(train_metrics)
            running_losses.clear()

            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(model, tokenizer, output_dir / f"checkpoint_step_{global_step:06d}")

            if valid_dataloader is not None and args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                val_loss = evaluate_validation_loss(model, valid_dataloader, args.device)
                eval_metrics = {
                    "eval_step": global_step,
                    "epoch": epoch + 1,
                    "eval/val_loss": val_loss,
                }
                append_jsonl(eval_metrics_path, eval_metrics)
                if wandb_run is not None:
                    wandb_run.log(eval_metrics)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, tokenizer, output_dir / "best_model")

    final_val_loss = evaluate_validation_loss(model, valid_dataloader, args.device)
    save_checkpoint(model, tokenizer, output_dir / "final_model")
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "train_path": str(args.train_path),
        "valid_path": str(args.valid_path),
        "seq_length": args.seq_length,
        "train_batch_size": args.train_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else final_val_loss,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.finish()
    return summary


def main() -> None:
    summary = train_alignment_sft(parse_args())
    logger.info("Alignment SFT summary: %s", json.dumps(summary, indent=2))


def cli() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()


if __name__ == "__main__":
    cli()
