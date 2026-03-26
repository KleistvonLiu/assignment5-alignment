from __future__ import annotations

import re
from statistics import mean
from typing import Any


MMLU_LETTER_PATTERN = re.compile(
    r"(?:correct answer is|answer is|answer|option|choice)\s*[:\-]?\s*[\(\[]?\s*([ABCD])\b",
    re.IGNORECASE,
)
MMLU_STANDALONE_LETTER_PATTERN = re.compile(r"\b([ABCD])\b")
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _option_text_regex(option_text: str) -> re.Pattern[str]:
    escaped = re.escape(_normalize_text(option_text))
    return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE)


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    if not model_output.strip():
        return None

    explicit_match = MMLU_LETTER_PATTERN.search(model_output)
    if explicit_match:
        return explicit_match.group(1).upper()

    standalone_letters = {
        match.group(1).upper() for match in MMLU_STANDALONE_LETTER_PATTERN.finditer(model_output)
    }
    if len(standalone_letters) == 1:
        return next(iter(standalone_letters))

    option_matches: list[str] = []
    for idx, option_text in enumerate(mmlu_example["options"]):
        if _option_text_regex(option_text).search(_normalize_text(model_output)):
            option_matches.append(chr(ord("A") + idx))
    if len(option_matches) == 1:
        return option_matches[0]
    return None


def normalize_numeric_string(number_str: str) -> str:
    normalized = number_str.replace(",", "")
    if normalized.startswith("+"):
        normalized = normalized[1:]
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = NUMBER_PATTERN.findall(model_output)
    if not matches:
        return None
    return normalize_numeric_string(matches[-1])


def extract_gsm8k_final_answer(answer: str) -> str:
    final_match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if final_match:
        return normalize_numeric_string(final_match.group(1))
    parsed = parse_gsm8k_response(answer)
    if parsed is None:
        raise ValueError(f"Could not extract GSM8K final answer from {answer!r}")
    return parsed


def summarize_binary_outcomes(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not records:
        return {"count": 0.0, f"{key}_mean": 0.0}
    values = [float(record[key]) for record in records]
    return {
        "count": float(len(values)),
        f"{key}_mean": mean(values),
    }


def summarize_length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {
            "avg_num_output_tokens": 0.0,
            "min_num_output_tokens": 0.0,
            "max_num_output_tokens": 0.0,
        }
    return {
        "avg_num_output_tokens": mean(lengths),
        "min_num_output_tokens": float(min(lengths)),
        "max_num_output_tokens": float(max(lengths)),
    }
