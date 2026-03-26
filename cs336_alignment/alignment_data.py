from __future__ import annotations

import csv
import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cs336_alignment.alignment_metrics import extract_gsm8k_final_answer
from cs336_alignment.alignment_prompts import render_alpaca_prompt


DEFAULT_SFT_TRAIN_PATH = Path(
    "/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
)
DEFAULT_SFT_VALID_PATH = Path(
    "/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz"
)
DEFAULT_HH_PATH = Path("/data/a5-alignment/hh")
HH_SPLIT_FILENAMES = (
    "harmless-base.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
)


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_sft_prompt_response(example: dict[str, Any]) -> tuple[str, str]:
    for prompt_key, response_key in (
        ("prompt", "response"),
        ("instruction", "output"),
        ("input", "output"),
        ("question", "answer"),
    ):
        prompt = example.get(prompt_key)
        response = example.get(response_key)
        if isinstance(prompt, str) and isinstance(response, str):
            return prompt, response

    messages = example.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        user_message = messages[0]
        assistant_message = messages[1]
        if (
            isinstance(user_message, dict)
            and isinstance(assistant_message, dict)
            and user_message.get("role") in {"user", "human"}
            and assistant_message.get("role") == "assistant"
        ):
            prompt = user_message.get("content")
            response = assistant_message.get("content")
            if isinstance(prompt, str) and isinstance(response, str):
                return prompt, response

    raise KeyError(
        "Could not extract a (prompt, response) pair from example keys "
        f"{sorted(example.keys())}"
    )


@dataclass(frozen=True)
class PackedSFTExample:
    input_ids: Tensor
    labels: Tensor


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be positive")

        self.seq_length = seq_length
        self.tokenizer = tokenizer

        documents = read_jsonl(dataset_path)
        rendered_documents: list[str] = []
        for example in documents:
            prompt, response = extract_sft_prompt_response(example)
            rendered_documents.append(render_alpaca_prompt(prompt, response).rstrip())

        if shuffle:
            random.shuffle(rendered_documents)

        all_token_ids: list[int] = []
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")

        for document in rendered_documents:
            token_ids = tokenizer(document, add_special_tokens=True)["input_ids"]
            all_token_ids.extend(token_ids)
            all_token_ids.append(eos_token_id)

        usable_token_count = ((len(all_token_ids) - 1) // seq_length) * seq_length
        self.examples: list[PackedSFTExample] = []
        for start in range(0, usable_token_count, seq_length):
            input_ids = torch.tensor(
                all_token_ids[start : start + seq_length],
                dtype=torch.long,
            )
            labels = torch.tensor(
                all_token_ids[start + 1 : start + seq_length + 1],
                dtype=torch.long,
            )
            self.examples.append(PackedSFTExample(input_ids=input_ids, labels=labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "labels": example.labels,
        }


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_mmlu_examples(data_root: str | Path) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    test_root = data_root / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"Missing MMLU test directory: {test_root}")

    examples: list[dict[str, Any]] = []
    for csv_path in sorted(test_root.glob("*_test.csv")):
        subject = csv_path.name.removesuffix("_test.csv")
        with csv_path.open(newline="") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) != 6:
                    raise ValueError(f"Expected 6 columns in {csv_path}, got {len(row)}")
                question, option_a, option_b, option_c, option_d, answer = row
                examples.append(
                    {
                        "example_id": f"{subject}:{row_idx}",
                        "subject": subject.replace("_", " "),
                        "question": question,
                        "options": [option_a, option_b, option_c, option_d],
                        "answer": answer,
                    }
                )
    return examples


def load_gsm8k_examples(path: str | Path) -> list[dict[str, Any]]:
    examples = read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        normalized.append(
            {
                "example_id": idx,
                "question": example["question"],
                "answer": example["answer"],
                "gold_numeric_answer": extract_gsm8k_final_answer(example["answer"]),
            }
        )
    return normalized


def load_alpaca_eval_examples(path: str | Path) -> list[dict[str, Any]]:
    examples = read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        normalized.append(
            {
                "example_id": idx,
                "instruction": example["instruction"],
                "dataset": example["dataset"],
                "reference_output": example.get("output"),
                "reference_generator": example.get("generator"),
            }
        )
    return normalized


def load_simple_safety_tests(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            rows.append({"example_id": row_idx, **row})
    return rows


HH_TURN_PATTERN = re.compile(r"(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|\Z)", re.DOTALL)


def parse_hh_conversation(conversation: str) -> list[tuple[str, str]]:
    turns = [
        (speaker.lower(), _normalize_whitespace(content))
        for speaker, content in HH_TURN_PATTERN.findall(conversation.strip())
    ]
    return [(speaker, content) for speaker, content in turns if content]


def _iter_hh_paths(path_or_dir: str | Path) -> Iterator[Path]:
    path = Path(path_or_dir)
    if path.is_file():
        yield path
        return
    if not path.exists():
        raise FileNotFoundError(f"HH data path does not exist: {path}")
    for filename in HH_SPLIT_FILENAMES:
        candidate = path / filename
        if candidate.exists():
            yield candidate


def load_hh_preferences(path_or_dir: str | Path = DEFAULT_HH_PATH) -> list[dict[str, Any]]:
    paths = list(_iter_hh_paths(path_or_dir))
    if not paths:
        raise FileNotFoundError(
            f"Could not find any HH files under {path_or_dir}; expected one of {HH_SPLIT_FILENAMES}"
        )

    normalized: list[dict[str, Any]] = []
    for path in paths:
        source_split = path.name.removesuffix(".jsonl.gz").removesuffix(".jsonl")
        for row_idx, example in enumerate(read_jsonl(path)):
            chosen_turns = parse_hh_conversation(example["chosen"])
            rejected_turns = parse_hh_conversation(example["rejected"])
            if len(chosen_turns) != 2 or len(rejected_turns) != 2:
                continue
            if chosen_turns[0][0] != "human" or chosen_turns[1][0] != "assistant":
                continue
            if rejected_turns[0][0] != "human" or rejected_turns[1][0] != "assistant":
                continue
            if chosen_turns[0][1] != rejected_turns[0][1]:
                continue
            normalized.append(
                {
                    "example_id": f"{source_split}:{row_idx}",
                    "instruction": chosen_turns[0][1],
                    "response_chosen": chosen_turns[1][1],
                    "response_rejected": rejected_turns[1][1],
                    "source_split": source_split,
                }
            )
    return normalized


def split_train_validation(
    examples: Sequence[dict[str, Any]],
    validation_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_size <= 0:
        return list(examples), []
    if validation_size >= len(examples):
        raise ValueError("validation_size must be smaller than the dataset size")
    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    validation_indices = set(indices[:validation_size])
    train_examples: list[dict[str, Any]] = []
    validation_examples: list[dict[str, Any]] = []
    for idx, example in enumerate(examples):
        if idx in validation_indices:
            validation_examples.append(example)
        else:
            train_examples.append(example)
    return train_examples, validation_examples
