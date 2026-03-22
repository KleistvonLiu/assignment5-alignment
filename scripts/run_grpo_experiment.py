#!/usr/bin/env python3
"""
Thin CLI wrapper for `cs336_alignment.grpo_experiment`.

uv run python scripts/run_grpo_experiment.py \
  --model-name-or-path model/Qwen2.5-Math-1.5B \
  --train-path data/MATHv2/train.jsonl \
  --validation-path data/MATHv2/validation.jsonl \
  --output-dir outputs/grpo_full_mathv2 \
  --n-grpo-steps 200 \
  --learning-rate 1e-5 \
  --advantage-eps 1e-6 \
  --rollout-batch-size 256 \
  --group-size 8 \
  --sampling-temperature 1.0 \
  --sampling-min-tokens 4 \
  --sampling-max-tokens 1024 \
  --epochs-per-rollout-batch 1 \
  --train-batch-size 256 \
  --gradient-accumulation-steps 128 \
  --loss-type reinforce_with_baseline \
  --policy-device cuda:0 \
  --vllm-device cuda:0 \
  --eval-every-steps 10 \
  --num-eval-examples 1024 \
  --num-log-generations 4 \
  --num-log-rollouts 4 \
  --gpu-memory-utilization 0.20
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_alignment.grpo_experiment import cli


if __name__ == "__main__":
    cli()
