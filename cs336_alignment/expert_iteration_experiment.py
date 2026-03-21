from __future__ import annotations

"""MATH 上的 Expert Iteration 训练脚本。

这份实现直接对应 handout 里的 Algorithm 2，核心流程是：

1. 从训练问题集合里采样一批题目。
2. 用当前 policy 通过 vLLM 为每道题采样多条回答。
3. 用 reward function 只保留答对且格式正确的 rollout。
4. 把这些“正确 rollout”当作新的 SFT 数据，对 policy 做若干轮监督微调。
5. 在验证集上评估，并记录生成熵等诊断指标。

与普通 SFT 的区别在于：这里的 SFT 数据不是外部固定给定的，而是由模型自己
在每一轮 EI 中生成，再经过 reward 过滤得到。

uv run python scripts/run_expert_iteration_experiment.py \
  --model-name-or-path model/Qwen2.5-Math-1.5B \
  --train-path data/MATHv2/train.jsonl \
  --validation-path data/MATHv2/validation.jsonl \
  --output-dir outputs/ei_g4_db512_ep1 \
  --n-ei-steps 5 \
  --question-batch-size 512 \
  --num-rollouts-per-question 4 \
  --sft-epochs-per-step 1 \
  --train-batch-size 64 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --max-grad-norm 1.0 \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --num-eval-examples 256 \
  --num-log-generations 4 \
  --gpu-memory-utilization 0.85 \
  --save-each-ei-step

"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from cs336_alignment.sft import get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.sft_experiment import (
    append_jsonl,
    build_sft_dataloader,
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
    """解析 Expert Iteration 实验所需的命令行参数。

    参数设计基本围绕三类超参：
    - rollout 阶段：每题采样多少条、生成长度、采样温度
    - SFT 阶段：每轮 EI 内训练几个 epoch、batch size、学习率
    - 评估阶段：每轮结束后评估多少 validation 样本，以及是否记录 generation 日志
    """
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
        help="Path to the EI question dataset.",
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
    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--question-batch-size", type=int, default=512)
    parser.add_argument("--num-rollouts-per-question", type=int, default=4)
    parser.add_argument("--sft-epochs-per-step", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-max-tokens", type=int, default=1024)
    parser.add_argument("--rollout-min-tokens", type=int, default=4)
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
    parser.add_argument("--num-eval-examples", type=int, default=256)
    parser.add_argument("--num-log-generations", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--save-each-ei-step", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def sample_question_batch(
    examples: list[dict[str, Any]],
    question_batch_size: int,
) -> list[dict[str, Any]]:
    """从训练题库中无放回采样一个问题批次 D_b。

    如果请求的 batch size 比数据集还大，就退化成取完整数据集。
    这里直接用 `random.sample`，因此每轮 EI 看到的是一个新的题目子集。
    """
    batch_size = min(question_batch_size, len(examples))
    return random.sample(examples, k=batch_size)


def validate_cuda_device_or_raise(device: str, *, arg_name: str) -> None:
    """在真正初始化模型前，先检查请求的 CUDA 设备是否存在。

    这样可以把原本由底层库抛出的 `invalid device ordinal`，
    提前变成更容易理解的参数错误。
    """
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


def truncate_after_first_answer_tag(response: str) -> str:
    """把生成结果裁到第一个 `</answer>` 为止。

    handout 明确要求 vLLM 在 `</answer>` 处停止，但实际生成时仍可能出现：
    - stop string 后又多带一点残留文本
    - 某些情况下 stop 没有完全生效

    为了让 reward 评估更稳定，这里再做一层保守裁剪。
    """
    stop_str = "</answer>"
    idx = response.find(stop_str)
    if idx == -1:
        return response
    return response[: idx + len(stop_str)]


def generate_rollouts(
    *,
    llm: LLM,
    prompt_template: str,
    question_examples: list[dict[str, Any]],
    num_rollouts_per_question: int,
    rollout_temperature: float,
    rollout_max_tokens: int,
    rollout_min_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    """对一批题目做 rollout，并为每条 rollout 计算奖励。

    返回的是“展开后”的 rollout 列表：
    - 如果有 B 道题、每题采样 G 条，那么最终返回 B * G 条记录
    - 每条记录都包含 prompt、response、ground_truth、reward 结果和原始题目信息

    这样后续过滤和落盘都更直接，不需要再维护嵌套结构。
    """
    from vllm import SamplingParams

    prompts = format_prompts(prompt_template, question_examples)
    ground_truths = [extract_ground_truth(example) for example in question_examples]
    # `min_tokens=4` 是 handout 特别建议的设置，
    # 目的是避免模型直接生成空串，导致下游 reward 或日志计算出 NaN。
    sampling_params = SamplingParams(
        temperature=rollout_temperature,
        max_tokens=rollout_max_tokens,
        min_tokens=rollout_min_tokens,
        n=num_rollouts_per_question,
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
            # 再做一次本地裁剪，确保 reward 看到的是完整且尽量干净的 `<answer>` 终止片段。
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


def filter_rollouts_for_sft(rollout_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只保留 reward=1 的 rollout，并转成 SFT 训练样本。

    这里使用最严格的标准：只有格式正确且最终答案正确的 rollout 才会进入下一轮 SFT。
    这样得到的数据质量更高，也更贴近“专家轨迹”的设定。
    """
    sft_examples: list[dict[str, Any]] = []
    for record in rollout_records:
        if float(record["metrics"]["reward"]) != 1.0:
            continue
        sft_examples.append(
            {
                "prompt": record["prompt"],
                "response": record["response"],
                "ground_truth": record["ground_truth"],
                "source_unique_id": record.get("unique_id"),
                "source_rollout_index": record["rollout_index"],
            }
        )
    return sft_examples


def run_sft_epochs_on_examples(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    sft_examples: list[dict[str, Any]],
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    policy_device: str,
    num_epochs: int,
    metrics_path: Path,
    ei_step: int,
    wandb_run: Any | None,
) -> None:
    """对某一轮 EI 过滤出的正确样本执行若干个 SFT epoch。

    这一段逻辑和前面单独实现的 SFT 训练步骤基本一致：
    - 先把 prompt/response tokenize 成 input_ids、labels、response_mask
    - 只在 response token 上计算 NLL
    - 支持 gradient accumulation
    - 每个 optimizer step 后记录 loss、entropy、grad norm

    注意这里每一轮 EI 都重新构造一个 optimizer。
    这样做更贴近“每步 EI 单独做一次小规模 SFT”的写法，也更容易分析每步 EI 的效果。
    """
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")
    micro_batch_size = train_batch_size // gradient_accumulation_steps

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        dataloader = build_sft_dataloader(
            examples=sft_examples,
            tokenizer=tokenizer,
            batch_size=micro_batch_size,
            shuffle=True,
        )
        num_microbatches = len(dataloader)
        if num_microbatches == 0:
            return
        epoch_iterator = tqdm(
            enumerate(dataloader),
            total=num_microbatches,
            desc=f"ei_step {ei_step} sft_epoch {epoch + 1}/{num_epochs}",
        )
        loss_window: list[float] = []
        entropy_window: list[float] = []
        optimizer_steps = 0
        for microbatch_idx, batch in epoch_iterator:
            if microbatch_idx % gradient_accumulation_steps == 0:
                # 最后一个 accumulation window 可能不足完整的 `gradient_accumulation_steps`，
                # 因此这里动态计算当前窗口大小，保证 loss 缩放正确。
                current_window_size = min(
                    gradient_accumulation_steps,
                    num_microbatches - microbatch_idx,
                )

            input_ids = batch["input_ids"].to(policy_device)
            labels = batch["labels"].to(policy_device)
            response_mask = batch["response_mask"].to(policy_device)
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
            loss_window.append(float(loss.item()))

            # 这里只记录 response token 上的平均熵，用来观察模型是否越来越“自信”。
            # prompt 部分不参与统计，因为它不属于模型本轮要学习的输出。
            response_lengths = response_mask.sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0)
            token_entropy = outputs["token_entropy"]
            avg_entropy = (
                (token_entropy * response_mask.to(dtype=token_entropy.dtype)).sum(dim=-1)
                / response_lengths
            ).mean()
            entropy_window.append(float(avg_entropy.item()))

            is_window_end = (
                (microbatch_idx + 1) % gradient_accumulation_steps == 0
                or microbatch_idx + 1 == num_microbatches
            )
            if not is_window_end:
                continue

            grad_norm = float(clip_grad_norm_(model.parameters(), max_grad_norm).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            metrics = {
                "ei_step": ei_step,
                "sft_epoch": epoch + 1,
                "optimizer_step_within_ei": optimizer_steps,
                "train/loss": float(sum(loss_window) / max(len(loss_window), 1)),
                "train/avg_token_entropy": float(sum(entropy_window) / max(len(entropy_window), 1)),
                "train/grad_norm": grad_norm,
            }
            append_jsonl(metrics_path, metrics)
            logger.info("expert iteration train metrics: %s", metrics)
            if wandb_run is not None:
                wandb_run.log(metrics)
            loss_window.clear()
            entropy_window.clear()


def maybe_plot_entropy(eval_metrics_path: Path, output_path: Path) -> None:
    """根据评估日志导出“响应熵随 EI step 变化”的曲线图。

    handout 要求绘制 entropy plot。这里优先使用 matplotlib；
    如果当前环境没有装 matplotlib，则退化成直接生成一张 SVG，
    这样依然能满足作业里“产出图”的要求，不会因为少一个依赖就卡住。
    """
    rows: list[dict[str, Any]] = []
    if not eval_metrics_path.exists():
        return
    with eval_metrics_path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        generation_summary = row.get("logged_generation_summary")
        if generation_summary is None:
            continue
        xs.append(int(row["ei_step"]))
        ys.append(float(generation_summary["avg_token_entropy"]))

    if not xs:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # 轻量级后备方案：直接手写一张 SVG 折线图。
        # 这样即使课程环境没有 matplotlib，也能稳定产出可提交的图文件。
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
  <text x="{width / 2:.2f}" y="20" font-size="18" text-anchor="middle" fill="#111">Expert Iteration Response Entropy</text>
  {' '.join(grid_elements)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.5" />
  <polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{polyline_points}" />
  {circle_elements}
  {' '.join(y_label_elements)}
  {x_label_elements}
  <text x="{width / 2:.2f}" y="{height - 12}" font-size="14" text-anchor="middle" fill="#111">EI Step</text>
  <text x="18" y="{height / 2:.2f}" font-size="14" text-anchor="middle" fill="#111" transform="rotate(-90 18 {height / 2:.2f})">Average Token Entropy</text>
</svg>
"""
        output_path.write_text(svg)
        logger.info("Saved entropy plot to %s", output_path)
        return

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("EI Step")
    plt.ylabel("Average Token Entropy")
    plt.title("Expert Iteration Response Entropy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved entropy plot to %s", output_path)


def train_expert_iteration(args: argparse.Namespace) -> None:
    """运行完整的 Expert Iteration 实验。

    整体流程：
    - 加载 train / validation / prompt / model / tokenizer
    - 初始化 policy 与独立的 vLLM 推理实例
    - 先做一次 step 0 验证，作为 EI 前的基线
    - 迭代执行：
      1. 采样问题
      2. rollout
      3. reward 过滤
      4. 基于过滤结果做 SFT
      5. 再次验证并记录指标
    - 保存最终模型和熵图
    """
    seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    validate_cuda_device_or_raise(args.policy_device, arg_name="policy_device")
    validate_cuda_device_or_raise(args.vllm_device, arg_name="vllm_device")

    # 课程机器上路径通常在 `/data/a5-alignment/...`；
    # 本地开发时则优先回退到仓库里的 `data/MATHv2/...`。
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

    train_examples = load_jsonl(train_path)
    validation_examples = load_validation_examples(validation_path, args.hendrycks_math_root)
    prompt_template = load_prompt_template(args.prompt_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_metrics_path = output_dir / "rollout_metrics.jsonl"
    train_metrics_path = output_dir / "train_metrics.jsonl"
    eval_metrics_path = output_dir / "eval_metrics.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if "cuda" in args.policy_device and torch.cuda.is_available() else torch.float32
    attn_implementation = "flash_attention_2" if "cuda" in args.policy_device and torch.cuda.is_available() else None
    # policy 模型是真正参与训练更新的那一份权重。
    policy = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    policy.to(args.policy_device)
    policy.train()

    wandb_run = setup_wandb(args)
    # 单独起一个 vLLM 实例做 rollout / validation 生成。
    # 这样可以避免直接用训练中的 policy 调 generate，吞吐更高，也更接近 handout 设定。
    llm = init_vllm(
        model_id=model_name_or_path,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 先做 EI 前的基线评估，相当于 step 0。
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
        wandb_run=wandb_run,
    )
    initial_eval["ei_step"] = 0
    append_jsonl(eval_metrics_path, initial_eval)

    for ei_step in range(1, args.n_ei_steps + 1):
        logger.info("starting expert iteration step %d/%d", ei_step, args.n_ei_steps)

        # Step 1: 采样这一轮要做 rollout 的题目子集 D_b。
        question_batch = sample_question_batch(train_examples, args.question_batch_size)

        # Step 2: 把当前 policy 权重同步进 vLLM，再进行 rollout。
        # 这样 rollout 使用的是“最新”的 policy，而不是初始化时的旧权重。
        load_policy_into_vllm_instance(policy, llm)
        rollout_records = generate_rollouts(
            llm=llm,
            prompt_template=prompt_template,
            question_examples=question_batch,
            num_rollouts_per_question=args.num_rollouts_per_question,
            rollout_temperature=args.rollout_temperature,
            rollout_max_tokens=args.rollout_max_tokens,
            rollout_min_tokens=args.rollout_min_tokens,
            seed=args.seed + ei_step,
        )

        # 把原始 rollout 全量落盘，后续分析格式错误、答案错误、长度分布都需要它。
        step_dir = output_dir / f"ei_step_{ei_step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = step_dir / "rollouts.jsonl"
        with rollout_path.open("w") as f:
            for record in rollout_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Step 3: 只保留 reward=1 的 rollout，作为下一阶段 SFT 数据。
        sft_examples = filter_rollouts_for_sft(rollout_records)
        filtered_path = step_dir / "filtered_sft.jsonl"
        with filtered_path.open("w") as f:
            for example in sft_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        # 记录这一轮 rollout 的总体统计，便于比较不同 G、不同 D_b 大小时的保留率。
        rewards = [float(record["metrics"]["reward"]) for record in rollout_records]
        rollout_summary = {
            "ei_step": ei_step,
            "num_questions": len(question_batch),
            "num_rollouts": len(rollout_records),
            "num_filtered_sft_examples": len(sft_examples),
            "avg_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
            "retention_rate": (len(sft_examples) / len(rollout_records)) if rollout_records else 0.0,
        }
        append_jsonl(rollout_metrics_path, rollout_summary)
        logger.info("expert iteration rollout summary: %s", rollout_summary)
        if wandb_run is not None:
            wandb_run.log({f"ei/{k}": v for k, v in rollout_summary.items() if k != "ei_step"} | {"ei_step": ei_step})

        if sft_examples:
            # Step 4: 在“当前 EI 轮收集到的正确样本”上做一小段 SFT。
            run_sft_epochs_on_examples(
                model=policy,
                tokenizer=tokenizer,
                sft_examples=sft_examples,
                train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                policy_device=args.policy_device,
                num_epochs=args.sft_epochs_per_step,
                metrics_path=train_metrics_path,
                ei_step=ei_step,
                wandb_run=wandb_run,
            )
        else:
            # 如果这一轮一个正确样本都没有，就只能跳过更新。
            # 这在 base model 很弱、rollout 配置又偏小时很常见。
            logger.warning("EI step %d produced no correct rollouts; skipping SFT update", ei_step)

        # Step 5: 每轮 EI 结束后重新评估，观察验证集表现和平均熵是否变化。
        eval_summary = evaluate_policy(
            policy=policy,
            tokenizer=tokenizer,
            llm=llm,
            validation_examples=validation_examples,
            prompt_template=prompt_template,
            model_name_or_path=model_name_or_path,
            output_dir=output_dir / "evals",
            eval_step=ei_step,
            num_eval_examples=args.num_eval_examples,
            num_log_generations=args.num_log_generations,
            wandb_run=wandb_run,
        )
        eval_summary["ei_step"] = ei_step
        append_jsonl(eval_metrics_path, eval_summary)

        if args.save_each_ei_step:
            save_checkpoint(policy, tokenizer, output_dir, f"checkpoint_ei_step_{ei_step:02d}")

    final_checkpoint = save_checkpoint(policy, tokenizer, output_dir, "final_model")
    # 作业要求输出 entropy 曲线，因此这里在训练结束后统一导出。
    maybe_plot_entropy(eval_metrics_path, output_dir / "entropy_over_training.svg")
    logger.info("Saved final expert iteration model to %s", final_checkpoint)
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    """CLI 主入口。"""
    args = parse_args()
    train_expert_iteration(args)


def cli() -> None:
    """带日志初始化的命令行入口。"""
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info("running %s", " ".join(sys.argv))
    main()
    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    cli()
