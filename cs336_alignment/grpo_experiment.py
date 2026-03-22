from __future__ import annotations

"""GRPO experiment loop for MATH.

This module assembles the previously implemented GRPO primitives into a full
training loop:

1. Sample a batch of questions.
2. Generate `group_size` rollouts per question with vLLM.
3. Score those rollouts with the reward function and compute advantages.
4. Reuse the rollout batch for one or more epochs of GRPO updates.
5. Periodically evaluate on the validation set and serialize diagnostics.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_math_baseline import (
    extract_ground_truth,
    format_prompts,
    load_prompt_template,
    load_validation_examples,
)
from cs336_alignment.generation_logging import log_generations
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean,
)
from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.sft_experiment import (
    append_jsonl,
    evaluate_policy,
    init_vllm,
    load_jsonl,
    load_policy_into_vllm_instance,
    resolve_path_with_local_fallback,
    save_checkpoint,
    seed_everything,
    setup_wandb,
)

if TYPE_CHECKING:
    from vllm import LLM

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
        default=Path("/data/a5-alignment/MATH/train.jsonl"),
        help="Path to the MATH train JSONL used for rollout questions.",
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
        help="Prompt template used for rollouts and validation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        default="reinforce_with_baseline",
    )
    parser.set_defaults(use_std_normalization=True)
    parser.add_argument(
        "--use-std-normalization",
        dest="use_std_normalization",
        action="store_true",
        help="Normalize advantages by per-group std.",
    )
    parser.add_argument(
        "--disable-std-normalization",
        dest="use_std_normalization",
        action="store_false",
        help="Use centered rewards without std normalization (Dr. GRPO style).",
    )
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
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--num-eval-examples", type=int, default=1024)
    parser.add_argument("--num-log-generations", type=int, default=4)
    parser.add_argument("--num-log-rollouts", type=int, default=4)
    parser.add_argument("--save-every-eval", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture inside vLLM for stability.",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def validate_cuda_device_or_raise(device: str, *, arg_name: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise ValueError(f"{arg_name}={device!r} was requested, but CUDA is unavailable.")
    if device == "cuda":
        return
    prefix = "cuda:"
    if not device.startswith(prefix):
        raise ValueError(f"{arg_name} must look like 'cuda' or 'cuda:N', got {device!r}.")
    try:
        device_index = int(device[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"{arg_name} must look like 'cuda' or 'cuda:N', got {device!r}.") from exc
    device_count = torch.cuda.device_count()
    if device_index < 0 or device_index >= device_count:
        raise ValueError(
            f"{arg_name}={device!r} is invalid on this machine: "
            f"only {device_count} CUDA device(s) are available."
        )


def _format_json_for_log(payload: dict[str, Any]) -> str:
    """把配置或统计字典格式化成更易读的日志字符串。"""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def sample_question_batch(
    examples: list[dict[str, Any]],
    question_batch_size: int,
) -> list[dict[str, Any]]:
    batch_size = min(question_batch_size, len(examples))
    return random.sample(examples, k=batch_size)


def truncate_after_first_answer_tag(response: str) -> str:
    stop_str = "</answer>"
    idx = response.find(stop_str)
    if idx == -1:
        return response
    return response[: idx + len(stop_str)]


def generate_rollout_records(
    *,
    llm: LLM,
    prompt_template: str,
    question_examples: list[dict[str, Any]],
    group_size: int,
    sampling_temperature: float,
    sampling_max_tokens: int,
    sampling_min_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    prompts = format_prompts(prompt_template, question_examples)
    ground_truths = [extract_ground_truth(example) for example in question_examples]
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    raw_outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    rollout_records: list[dict[str, Any]] = []
    for question_idx, (example, prompt, ground_truth, raw_output) in enumerate(
        zip(question_examples, prompts, ground_truths, raw_outputs)
    ):
        for rollout_idx, generation in enumerate(raw_output.outputs):
            response = truncate_after_first_answer_tag(generation.text)
            metrics = r1_zero_reward_fn(response, ground_truth)
            rollout_records.append(
                {
                    "question_index": question_idx,
                    "rollout_index": rollout_idx,
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": ground_truth,
                    "metrics": metrics,
                    "num_output_tokens": len(getattr(generation, "token_ids", [])),
                    **example,
                }
            )
    return rollout_records


def score_model_log_probs_batched(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    logger.info(
        "开始分批打分 log-probs：样本数=%d，batch_size=%d，return_token_entropy=%s",
        int(input_ids.shape[0]),
        int(batch_size),
        bool(return_token_entropy),
    )

    was_training = model.training
    model.eval()
    log_prob_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for start in range(0, input_ids.shape[0], batch_size):
                end = min(start + batch_size, input_ids.shape[0])
                outputs = get_response_log_probs(
                    model=model,
                    input_ids=input_ids[start:end],
                    labels=labels[start:end],
                    return_token_entropy=return_token_entropy,
                )
                log_prob_chunks.append(outputs["log_probs"].cpu())
                if return_token_entropy:
                    entropy_chunks.append(outputs["token_entropy"].cpu())
    finally:
        if was_training:
            model.train()

    result = {"log_probs": torch.cat(log_prob_chunks, dim=0)}
    if return_token_entropy:
        result["token_entropy"] = torch.cat(entropy_chunks, dim=0)
    logger.info(
        "完成 log-probs 打分：log_probs 形状=%s%s",
        tuple(result["log_probs"].shape),
        f"，token_entropy 形状={tuple(result['token_entropy'].shape)}"
        if "token_entropy" in result
        else "",
    )
    return result


def summarize_rollout_records(
    rollout_records: list[dict[str, Any]],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
) -> dict[str, Any]:
    format_rewards = [float(record["metrics"]["format_reward"]) for record in rollout_records]
    answer_rewards = [float(record["metrics"]["answer_reward"]) for record in rollout_records]
    response_lengths = [float(record.get("num_output_tokens", 0)) for record in rollout_records]
    return {
        "num_rollouts": len(rollout_records),
        "reward_mean": float(raw_rewards.mean().item()) if raw_rewards.numel() else 0.0,
        "reward_std": float(raw_rewards.std(unbiased=False).item()) if raw_rewards.numel() > 1 else 0.0,
        "format_reward_mean": (sum(format_rewards) / len(format_rewards)) if format_rewards else 0.0,
        "answer_reward_mean": (sum(answer_rewards) / len(answer_rewards)) if answer_rewards else 0.0,
        "advantage_mean": float(advantages.mean().item()) if advantages.numel() else 0.0,
        "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        "avg_response_length": (sum(response_lengths) / len(response_lengths)) if response_lengths else 0.0,
    }


def maybe_log_rollout_examples(
    *,
    policy: torch.nn.Module,
    tokenizer: Any,
    rollout_records: list[dict[str, Any]],
    num_log_rollouts: int,
    rollout_step: int,
    wandb_run: Any | None,
) -> dict[str, Any] | None:
    if num_log_rollouts <= 0 or not rollout_records:
        return None
    logger.info(
        "开始记录 rollout 示例：rollout_step=%d，记录条数=%d",
        int(rollout_step),
        min(int(num_log_rollouts), len(rollout_records)),
    )
    records = rollout_records[:num_log_rollouts]
    log_output = log_generations(
        model=policy,
        tokenizer=tokenizer,
        prompts=[record["prompt"] for record in records],
        ground_truths=[record["ground_truth"] for record in records],
        responses=[record["response"] for record in records],
        reward_fn=r1_zero_reward_fn,
        log_prefix="rollout",
        step=rollout_step,
        log_example_details=False,
        wandb_run=wandb_run,
    )
    logger.info(
        "完成 rollout 示例记录：rollout_step=%d，摘要=%s",
        int(rollout_step),
        _format_json_for_log(log_output["summary"]),
    )
    return log_output["summary"]


def maybe_plot_series(
    *,
    xs: list[int],
    ys: list[float],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    if not xs:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        width = 640
        height = 400
        margin_left = 70
        margin_right = 24
        margin_top = 32
        margin_bottom = 54
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        def scale_x(x: int) -> float:
            if max_x == min_x:
                return margin_left + plot_width / 2.0
            return margin_left + (x - min_x) * plot_width / (max_x - min_x)

        def scale_y(y: float) -> float:
            if max_y == min_y:
                return margin_top + plot_height / 2.0
            return margin_top + (max_y - y) * plot_height / (max_y - min_y)

        polyline_points = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(xs, ys))
        circle_elements = "\n".join(
            f'<circle cx="{scale_x(x):.2f}" cy="{scale_y(y):.2f}" r="4" fill="#1f77b4" />'
            for x, y in zip(xs, ys)
        )
        x_label_elements = "\n".join(
            f'<text x="{scale_x(x):.2f}" y="{height - margin_bottom + 20}" '
            f'font-size="12" text-anchor="middle" fill="#333">{x}</text>'
            for x in xs
        )
        y_ticks = 5
        y_label_elements: list[str] = []
        grid_elements: list[str] = []
        for tick_idx in range(y_ticks):
            if y_ticks == 1:
                y_value = min_y
            else:
                y_value = min_y + (max_y - min_y) * tick_idx / (y_ticks - 1)
            y_pos = scale_y(y_value)
            y_label_elements.append(
                f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" '
                f'font-size="12" text-anchor="end" fill="#333">{y_value:.3f}</text>'
            )
            grid_elements.append(
                f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{width - margin_right}" y2="{y_pos:.2f}" '
                'stroke="#ddd" stroke-width="1" />'
            )

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="white" />
  <text x="{width / 2:.2f}" y="20" font-size="18" text-anchor="middle" fill="#111">{title}</text>
  {' '.join(grid_elements)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5" />
  <polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{polyline_points}" />
  {circle_elements}
  {' '.join(y_label_elements)}
  {x_label_elements}
  <text x="{width / 2:.2f}" y="{height - 12}" font-size="14" text-anchor="middle" fill="#111">{x_label}</text>
  <text x="18" y="{height / 2:.2f}" font-size="14" text-anchor="middle" fill="#111" transform="rotate(-90 18 {height / 2:.2f})">{y_label}</text>
</svg>
"""
        output_path.write_text(svg)
        logger.info("Saved plot to %s", output_path)
        return

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved plot to %s", output_path)


def maybe_plot_eval_curves(eval_metrics_path: Path, output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    if not eval_metrics_path.exists():
        logger.info("未找到评估日志 %s，跳过绘图", eval_metrics_path)
        return
    with eval_metrics_path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    reward_xs: list[int] = []
    reward_ys: list[float] = []
    entropy_xs: list[int] = []
    entropy_ys: list[float] = []
    for row in rows:
        train_step = int(row.get("train_step", row.get("eval_step", 0)))
        reward_mean = row.get("reward_mean")
        if reward_mean is not None:
            reward_xs.append(train_step)
            reward_ys.append(float(reward_mean))
        generation_summary = row.get("logged_generation_summary")
        if generation_summary is not None and generation_summary.get("avg_token_entropy") is not None:
            entropy_xs.append(train_step)
            entropy_ys.append(float(generation_summary["avg_token_entropy"]))

    maybe_plot_series(
        xs=reward_xs,
        ys=reward_ys,
        output_path=output_dir / "validation_reward_over_steps.svg",
        title="Validation Reward Over Train Steps",
        x_label="Train Step",
        y_label="Validation Reward Mean",
    )
    maybe_plot_series(
        xs=entropy_xs,
        ys=entropy_ys,
        output_path=output_dir / "validation_entropy_over_steps.svg",
        title="Validation Token Entropy Over Train Steps",
        x_label="Train Step",
        y_label="Validation Avg Token Entropy",
    )


def train_grpo(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    validate_cuda_device_or_raise(args.policy_device, arg_name="policy_device")
    validate_cuda_device_or_raise(args.vllm_device, arg_name="vllm_device")

    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if args.train_batch_size < args.group_size:
        raise ValueError("train_batch_size must be greater than or equal to group_size")

    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size

    logger.info(
        "开始 GRPO 训练，关键超参数如下：\n%s",
        _format_json_for_log(
            {
                "n_grpo_steps": args.n_grpo_steps,
                "learning_rate": args.learning_rate,
                "advantage_eps": args.advantage_eps,
                "rollout_batch_size": args.rollout_batch_size,
                "group_size": args.group_size,
                "n_prompts_per_rollout_batch": n_prompts_per_rollout_batch,
                "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "micro_train_batch_size": micro_train_batch_size,
                "loss_type": args.loss_type,
                "use_std_normalization": args.use_std_normalization,
                "cliprange": args.cliprange,
                "sampling_temperature": args.sampling_temperature,
                "sampling_min_tokens": args.sampling_min_tokens,
                "sampling_max_tokens": args.sampling_max_tokens,
                "policy_device": args.policy_device,
                "vllm_device": args.vllm_device,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "vllm_enforce_eager": args.vllm_enforce_eager,
                "eval_every_steps": args.eval_every_steps,
                "num_eval_examples": args.num_eval_examples,
                "num_log_generations": args.num_log_generations,
                "num_log_rollouts": args.num_log_rollouts,
                "seed": args.seed,
            }
        ),
    )

    train_path = resolve_path_with_local_fallback(
        args.train_path,
        local_fallback=Path("data/MATHv2/train.jsonl"),
    )
    validation_path = resolve_path_with_local_fallback(
        args.validation_path,
        local_fallback=Path("data/MATHv2/validation.jsonl"),
    )
    model_name_or_path = args.model_name_or_path
    if not Path(model_name_or_path).exists():
        local_model_path = Path("model/Qwen2.5-Math-1.5B")
        if local_model_path.exists():
            model_name_or_path = str(local_model_path)

    logger.info(
        "路径解析完成：\n%s",
        _format_json_for_log(
            {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "prompt_path": str(args.prompt_path),
                "model_name_or_path": str(model_name_or_path),
                "output_dir": str(args.output_dir),
            }
        ),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_metrics_path = output_dir / "rollout_metrics.jsonl"
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"

    train_examples = load_jsonl(train_path)
    validation_examples = load_validation_examples(validation_path, args.hendrycks_math_root)
    prompt_template = load_prompt_template(args.prompt_path)
    logger.info(
        "数据加载完成：训练题目数=%d，验证题目数=%d，prompt 模板长度=%d 字符",
        len(train_examples),
        len(validation_examples),
        len(prompt_template),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(
        "Tokenizer 已加载：pad_token_id=%s，eos_token_id=%s",
        str(tokenizer.pad_token_id),
        str(tokenizer.eos_token_id),
    )

    torch_dtype = torch.bfloat16 if "cuda" in args.policy_device and torch.cuda.is_available() else torch.float32
    attn_implementation = "flash_attention_2" if "cuda" in args.policy_device and torch.cuda.is_available() else None
    policy = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    policy.to(args.policy_device)
    policy.train()
    logger.info(
        "Policy 模型已加载并切到训练模式：dtype=%s，attn_implementation=%s，device=%s",
        str(torch_dtype),
        str(attn_implementation),
        str(args.policy_device),
    )

    optimizer = AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    logger.info("优化器已初始化：AdamW(lr=%s, betas=(0.9, 0.95), weight_decay=0.0)", args.learning_rate)
    wandb_run = setup_wandb(args)
    if wandb_run is None:
        logger.info("未启用 wandb 日志")
    else:
        logger.info("已启用 wandb：project=%s，run_name=%s", args.wandb_project, args.wandb_run_name)

    logger.info(
        "开始初始化 vLLM：model=%s，device=%s，gpu_memory_utilization=%.3f",
        model_name_or_path,
        args.vllm_device,
        args.gpu_memory_utilization,
    )
    llm = init_vllm(
        model_id=model_name_or_path,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.vllm_enforce_eager,
    )
    logger.info("vLLM 初始化完成")

    global_train_step = 0
    last_eval_step = -1
    if args.num_eval_examples > 0:
        logger.info(
            "开始 step 0 验证：num_eval_examples=%d，num_log_generations=%d",
            args.num_eval_examples,
            args.num_log_generations,
        )
        initial_eval = evaluate_policy(
            policy=policy,
            tokenizer=tokenizer,
            llm=llm,
            validation_examples=validation_examples,
            prompt_template=prompt_template,
            model_name_or_path=model_name_or_path,
            output_dir=output_dir / "evals",
            eval_step=0,
            num_eval_examples=args.num_eval_examples,
            num_log_generations=args.num_log_generations,
            log_example_details=False,
            wandb_run=wandb_run,
        )
        initial_eval["train_step"] = 0
        initial_eval["rollout_step"] = 0
        append_jsonl(eval_metrics_path, initial_eval)
        last_eval_step = 0
        logger.info("step 0 验证完成：\n%s", _format_json_for_log(initial_eval))

    for rollout_step in range(1, args.n_grpo_steps + 1):
        logger.info("开始 GRPO rollout step %d/%d", rollout_step, args.n_grpo_steps)
        question_batch = sample_question_batch(train_examples, n_prompts_per_rollout_batch)
        logger.info(
            "已采样问题批次：题目数=%d，示例 unique_id 前3条=%s",
            len(question_batch),
            [
                example.get("unique_id", example.get("problem", "")[:40])
                for example in question_batch[:3]
            ],
        )
        logger.info("将最新 policy 权重同步到 vLLM 实例")
        load_policy_into_vllm_instance(policy, llm)
        logger.info(
            "开始 rollout 采样：group_size=%d，temperature=%.3f，max_tokens=%d，min_tokens=%d",
            args.group_size,
            args.sampling_temperature,
            args.sampling_max_tokens,
            args.sampling_min_tokens,
        )
        rollout_records = generate_rollout_records(
            llm=llm,
            prompt_template=prompt_template,
            question_examples=question_batch,
            group_size=args.group_size,
            sampling_temperature=args.sampling_temperature,
            sampling_max_tokens=args.sampling_max_tokens,
            sampling_min_tokens=args.sampling_min_tokens,
            seed=args.seed + rollout_step,
        )
        logger.info(
            "rollout 采样完成：共生成 %d 条回答（每题 %d 条）",
            len(rollout_records),
            args.group_size,
        )

        step_dir = output_dir / f"rollout_step_{rollout_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = step_dir / "rollouts.jsonl"
        with rollout_path.open("w") as f:
            for record in rollout_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("原始 rollout 已保存到 %s", rollout_path)

        prompts = [record["prompt"] for record in rollout_records]
        responses = [record["response"] for record in rollout_records]
        ground_truths = [record["ground_truth"] for record in rollout_records]
        logger.info("开始计算 raw rewards 与 group-normalized advantages")
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=ground_truths,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )
        logger.info(
            "奖励与 advantage 计算完成：\n%s",
            _format_json_for_log(
                {
                    "reward_metadata": reward_metadata,
                    "raw_reward_mean": float(raw_rewards.mean().item()) if raw_rewards.numel() else 0.0,
                    "raw_reward_std": float(raw_rewards.std(unbiased=False).item()) if raw_rewards.numel() > 1 else 0.0,
                    "advantage_mean": float(advantages.mean().item()) if advantages.numel() else 0.0,
                    "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
                }
            ),
        )

        logger.info("开始把 rollout prompt/response tokenize 成训练张量")
        tokenized = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer,
        )
        rollout_batch: dict[str, Any] = {
            "input_ids": tokenized["input_ids"].cpu(),
            "labels": tokenized["labels"].cpu(),
            "response_mask": tokenized["response_mask"].cpu(),
            "raw_rewards": raw_rewards.unsqueeze(-1).cpu(),
            "advantages": advantages.unsqueeze(-1).cpu(),
        }
        logger.info(
            "tokenize 完成：input_ids=%s，labels=%s，response_mask=%s",
            tuple(rollout_batch["input_ids"].shape),
            tuple(rollout_batch["labels"].shape),
            tuple(rollout_batch["response_mask"].shape),
        )
        if args.loss_type == "grpo_clip":
            logger.info("当前使用 grpo_clip，开始预计算并缓存 old_log_probs")
            old_log_prob_outputs = score_model_log_probs_batched(
                model=policy,
                input_ids=rollout_batch["input_ids"],
                labels=rollout_batch["labels"],
                batch_size=micro_train_batch_size,
                return_token_entropy=False,
            )
            rollout_batch["old_log_probs"] = old_log_prob_outputs["log_probs"].cpu()
            logger.info(
                "old_log_probs 缓存完成：形状=%s",
                tuple(rollout_batch["old_log_probs"].shape),
            )
        else:
            logger.info("当前 loss_type=%s，不需要 old_log_probs", args.loss_type)

        rollout_summary = {
            "rollout_step": rollout_step,
            "train_step": global_train_step,
            "num_questions": len(question_batch),
            "group_size": args.group_size,
            **reward_metadata,
            **summarize_rollout_records(
                rollout_records=rollout_records,
                raw_rewards=raw_rewards,
                advantages=advantages,
            ),
        }
        rollout_log_summary = maybe_log_rollout_examples(
            policy=policy,
            tokenizer=tokenizer,
            rollout_records=rollout_records,
            num_log_rollouts=args.num_log_rollouts,
            rollout_step=rollout_step,
            wandb_run=wandb_run,
        )
        if rollout_log_summary is not None:
            rollout_summary["logged_rollout_summary"] = rollout_log_summary
        append_jsonl(rollout_metrics_path, rollout_summary)
        logger.info("GRPO rollout summary: %s", rollout_summary)
        if wandb_run is not None:
            wandb_payload = {
                "rollout_step": rollout_step,
                "rollout/reward_mean": rollout_summary["reward_mean"],
                "rollout/reward_std": rollout_summary["reward_std"],
                "rollout/format_reward_mean": rollout_summary["format_reward_mean"],
                "rollout/answer_reward_mean": rollout_summary["answer_reward_mean"],
                "rollout/avg_response_length": rollout_summary["avg_response_length"],
            }
            wandb_run.log(wandb_payload)
            logger.info("已将 rollout 摘要写入 wandb")

        optimizer.zero_grad(set_to_none=True)
        num_examples = rollout_batch["input_ids"].shape[0]
        logger.info(
            "开始基于当前 rollout batch 做策略更新：num_examples=%d，epochs_per_rollout_batch=%d，micro_train_batch_size=%d",
            num_examples,
            args.epochs_per_rollout_batch,
            micro_train_batch_size,
        )
        for epoch in range(args.epochs_per_rollout_batch):
            permutation = torch.randperm(num_examples)
            num_microbatches = (num_examples + micro_train_batch_size - 1) // micro_train_batch_size
            logger.info(
                "进入 rollout_step=%d 的第 %d/%d 个训练 epoch：num_microbatches=%d",
                rollout_step,
                epoch + 1,
                args.epochs_per_rollout_batch,
                num_microbatches,
            )
            epoch_iterator = tqdm(
                range(num_microbatches),
                total=num_microbatches,
                desc=f"rollout {rollout_step}/{args.n_grpo_steps} epoch {epoch + 1}/{args.epochs_per_rollout_batch}",
            )
            loss_window: list[float] = []
            entropy_window: list[float] = []
            clip_fraction_window: list[float] = []
            for microbatch_idx in epoch_iterator:
                start = microbatch_idx * micro_train_batch_size
                end = min(start + micro_train_batch_size, num_examples)
                batch_indices = permutation[start:end]

                if microbatch_idx % args.gradient_accumulation_steps == 0:
                    current_window_size = min(
                        args.gradient_accumulation_steps,
                        num_microbatches - microbatch_idx,
                    )
                    logger.info(
                        "开启新的梯度累积窗口：rollout_step=%d，epoch=%d，microbatch_idx=%d，window_size=%d",
                        rollout_step,
                        epoch + 1,
                        microbatch_idx,
                        current_window_size,
                    )

                input_ids = rollout_batch["input_ids"][batch_indices].to(args.policy_device)
                labels = rollout_batch["labels"][batch_indices].to(args.policy_device)
                response_mask = rollout_batch["response_mask"][batch_indices].to(args.policy_device)
                raw_rewards_batch = rollout_batch["raw_rewards"][batch_indices].to(args.policy_device)
                advantages_batch = rollout_batch["advantages"][batch_indices].to(args.policy_device)
                old_log_probs_batch = None
                if args.loss_type == "grpo_clip":
                    old_log_probs_batch = rollout_batch["old_log_probs"][batch_indices].to(args.policy_device)

                outputs = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True,
                )
                loss, loss_metadata = grpo_microbatch_train_step(
                    policy_log_probs=outputs["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=current_window_size,
                    loss_type=args.loss_type,
                    raw_rewards=raw_rewards_batch,
                    advantages=advantages_batch,
                    old_log_probs=old_log_probs_batch,
                    cliprange=args.cliprange,
                )
                loss_window.append(float(loss.item()))

                avg_entropy = masked_mean(
                    tensor=outputs["token_entropy"],
                    mask=response_mask,
                    dim=None,
                )
                entropy_window.append(float(avg_entropy.item()))

                if "used_clipped" in loss_metadata:
                    clip_fraction = masked_mean(
                        tensor=loss_metadata["used_clipped"].to(dtype=outputs["log_probs"].dtype),
                        mask=response_mask,
                        dim=None,
                    )
                    clip_fraction_window.append(float(clip_fraction.item()))

                is_window_end = (
                    (microbatch_idx + 1) % args.gradient_accumulation_steps == 0
                    or microbatch_idx + 1 == num_microbatches
                )
                if not is_window_end:
                    continue

                grad_norm = float(clip_grad_norm_(policy.parameters(), args.max_grad_norm).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_train_step += 1

                train_metrics: dict[str, Any] = {
                    "train_step": global_train_step,
                    "rollout_step": rollout_step,
                    "epoch_within_rollout": epoch + 1,
                    "train/loss": float(sum(loss_window) / max(len(loss_window), 1)),
                    "train/avg_token_entropy": float(sum(entropy_window) / max(len(entropy_window), 1)),
                    "train/grad_norm": grad_norm,
                    "train/raw_reward_mean": float(raw_rewards_batch.mean().item()),
                    "train/advantage_mean": float(advantages_batch.mean().item()),
                }
                if clip_fraction_window:
                    train_metrics["train/clip_fraction"] = float(
                        sum(clip_fraction_window) / max(len(clip_fraction_window), 1)
                    )

                append_jsonl(train_metrics_path, train_metrics)
                logger.info("GRPO train step %d 指标：%s", global_train_step, _format_json_for_log(train_metrics))
                if wandb_run is not None:
                    wandb_run.log(train_metrics)
                    logger.info("已将 train step %d 指标写入 wandb", global_train_step)

                loss_window.clear()
                entropy_window.clear()
                clip_fraction_window.clear()

                if (
                    args.num_eval_examples > 0
                    and args.eval_every_steps > 0
                    and global_train_step % args.eval_every_steps == 0
                ):
                    logger.info(
                        "触发周期性验证：train_step=%d，num_eval_examples=%d",
                        global_train_step,
                        args.num_eval_examples,
                    )
                    eval_summary = evaluate_policy(
                        policy=policy,
                        tokenizer=tokenizer,
                        llm=llm,
                        validation_examples=validation_examples,
                        prompt_template=prompt_template,
                        model_name_or_path=model_name_or_path,
                        output_dir=output_dir / "evals",
                        eval_step=global_train_step,
                        num_eval_examples=args.num_eval_examples,
                        num_log_generations=args.num_log_generations,
                        log_example_details=False,
                        wandb_run=wandb_run,
                    )
                    eval_summary["train_step"] = global_train_step
                    eval_summary["rollout_step"] = rollout_step
                    append_jsonl(eval_metrics_path, eval_summary)
                    last_eval_step = global_train_step
                    logger.info("周期性验证完成：\n%s", _format_json_for_log(eval_summary))
                    if args.save_every_eval:
                        save_checkpoint(policy, tokenizer, output_dir, f"checkpoint_step_{global_train_step:06d}")
                        logger.info("已保存 checkpoint_step_%06d", global_train_step)

    if args.num_eval_examples > 0 and global_train_step != last_eval_step:
        logger.info(
            "训练结束后补做最后一次验证：train_step=%d，last_eval_step=%d",
            global_train_step,
            last_eval_step,
        )
        final_eval = evaluate_policy(
            policy=policy,
            tokenizer=tokenizer,
            llm=llm,
            validation_examples=validation_examples,
            prompt_template=prompt_template,
            model_name_or_path=model_name_or_path,
            output_dir=output_dir / "evals",
            eval_step=global_train_step,
            num_eval_examples=args.num_eval_examples,
            num_log_generations=args.num_log_generations,
            log_example_details=False,
            wandb_run=wandb_run,
        )
        final_eval["train_step"] = global_train_step
        final_eval["rollout_step"] = args.n_grpo_steps
        append_jsonl(eval_metrics_path, final_eval)
        logger.info("最终验证完成：\n%s", _format_json_for_log(final_eval))

    final_checkpoint = save_checkpoint(policy, tokenizer, output_dir, "final_model")
    logger.info("最终模型已保存到 %s", final_checkpoint)
    logger.info("开始导出验证 reward / entropy 曲线")
    maybe_plot_eval_curves(eval_metrics_path, output_dir)
    logger.info("GRPO 训练全部完成")
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("wandb run 已结束")


def main() -> None:
    args = parse_args()
    train_grpo(args)


def cli() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("开始运行命令：%s", " ".join(sys.argv))
    main()
    logger.info("命令运行结束：%s", sys.argv[0])


if __name__ == "__main__":
    cli()
