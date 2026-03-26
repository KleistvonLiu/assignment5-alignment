# CS336 Supplement Assignment Guide

This repository now includes the main code paths needed for the optional safety / SFT / DPO supplement without assuming that large models or cluster datasets are locally runnable.

## Scripts and Outputs

### Zero-shot / evaluation scripts
- `scripts/evaluate_mmlu.py`
  - Writes `results.jsonl` and `summary.json`.
  - Key summary fields: `accuracy`, `num_parse_failures`, `examples_per_second`.
- `scripts/evaluate_gsm8k.py`
  - Writes `results.jsonl` and `summary.json`.
  - Key summary fields: `accuracy`, `num_parse_failures`, `examples_per_second`.
- `scripts/generate_alpaca_eval_outputs.py`
  - Writes AlpacaEval-compatible JSON array plus `summary.json`.
  - Run AlpacaEval separately with:
    - `uv run alpaca_eval --model_outputs <path_to_json> --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' --base-dir '.'`
- `scripts/generate_safety_outputs.py`
  - Writes evaluator-compatible JSONL plus `summary.json`.
  - Run the safety evaluator separately with:
    - `uv run python scripts/evaluate_safety.py --input-path <path_to_jsonl> --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct --num-gpus 2 --output-path <path_to_eval_jsonl>`

### Training scripts
- `scripts/run_alignment_sft.py`
  - Packed next-token SFT on Alpaca-formatted prompt/response documents.
  - Writes `train_metrics.jsonl`, optional `eval_metrics.jsonl`, `summary.json`, `best_model/`, `final_model/`.
- `scripts/run_dpo_training.py`
  - DPO on single-turn Anthropic HH preference pairs.
  - Writes `train_metrics.jsonl`, `eval_metrics.jsonl`, `summary.json`, `best_model/`, `final_model/`.

## Static Answers You Can Write Without Running Models

### Prompt design
- Zero-shot MMLU wraps a benchmark-specific multiple-choice instruction inside the provided system prompt and expects a letter-valued answer.
- Zero-shot GSM8K wraps a short arithmetic question ending with `Answer:`.
- AlpacaEval and SimpleSafetyTests can be run either as raw instructions or with the Alpaca SFT prompt format when evaluating a fine-tuned model.
- SFT and DPO both use the Alpaca prompt template so that training and post-training scoring stay aligned.

### Parsing strategy
- MMLU parsing first looks for an explicit answer letter such as `The correct answer is B`, then falls back to exact option-text matches when the letter is absent.
- GSM8K parsing takes the final numeric span in the model output and normalizes commas and trivial decimal formatting.

### SFT data expectations
- The packed SFT dataset renders each `(prompt, response)` pair with the Alpaca template, trims the template's trailing document newline, appends EOS, concatenates documents, and slices the token stream into fixed-length next-token training examples.
- Packing minimizes padding and matches the assignment description of non-overlapping constant-length chunks.

### HH preference data observations
- The chosen HH response is usually more directly helpful, more cautious around risky requests, or more cooperative in tone.
- The rejected response is often evasive, less complete, less safe, or more willing to comply with harmful requests.
- A reasonable expectation is that the harmless subsets emphasize refusal style and risk mitigation, while the helpful subsets emphasize relevance, clarity, and instruction following.

### Red-teaming misuse ideas
- Targeted phishing or social engineering assistance.
- Malware authoring, obfuscation, or operational troubleshooting.
- Scalable disinformation, impersonation, or persuasion at volume.

## Run-Dependent Answers That Still Need Real Experiments

Do not invent numeric answers for the following. Pull them from the output files after a real run.

### Throughput
- Read `examples_per_second` from the relevant `summary.json`.

### Accuracy / parse failures
- MMLU and GSM8K:
  - Read `accuracy` and `num_parse_failures` from `summary.json`.
  - Inspect `results.jsonl` rows with `parse_failed=true` or `is_correct=false`.

### AlpacaEval winrate
- Read AlpacaEval's output artifacts after running the evaluator command.
- Compare baseline, SFT, and DPO runs using the same annotation setup.

### Safety rate
- Read the `safe` metric from the output of `scripts/evaluate_safety.py`.
- Inspect rows where `metrics.safe == 0.0` for unsafe examples.

## Trend Templates for Non-Executable Discussion

Use these only when you need a clearly marked non-empirical placeholder.

- SFT trend:
  - "Instruction tuning is expected to improve adherence to requested format and generally improve assistant-style responses, while potentially introducing a mild alignment tax on raw benchmark capability."
- DPO trend:
  - "DPO on HH is expected to further improve preference-model winrate and refusal behavior, especially on harmful prompts, but may slightly reduce raw MMLU/GSM8K performance if the policy becomes more conservative."
- Zero-shot baseline trend:
  - "A base model can answer many prompts, but it will usually show weaker formatting consistency, weaker refusal behavior, and lower pairwise preference winrate than an SFT or DPO model."

## Suggested Commands

### Zero-shot baseline examples
```bash
uv run python scripts/evaluate_mmlu.py \
  --model-name-or-path /data/a5-alignment/models/Llama-3.1-8B \
  --prompt-style zero_shot \
  --output-dir outputs/mmlu_baseline

uv run python scripts/evaluate_gsm8k.py \
  --model-name-or-path /data/a5-alignment/models/Llama-3.1-8B \
  --prompt-style zero_shot \
  --output-dir outputs/gsm8k_baseline
```

### SFT-style evaluation examples
```bash
uv run python scripts/evaluate_mmlu.py \
  --model-name-or-path outputs/alignment_sft/final_model \
  --prompt-style alpaca_sft \
  --output-dir outputs/mmlu_sft

uv run python scripts/generate_safety_outputs.py \
  --model-name-or-path outputs/alignment_sft/final_model \
  --prompt-style alpaca_sft \
  --output-path outputs/safety_sft/predictions.jsonl
```
