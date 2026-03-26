from __future__ import annotations

from enum import StrEnum
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ZERO_SHOT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "zero_shot_system_prompt.prompt"
ALPACA_SFT_PROMPT_PATH = PROMPTS_DIR / "alpaca_sft.prompt"
QUESTION_ONLY_PROMPT_PATH = PROMPTS_DIR / "question_only.prompt"


class PromptStyle(StrEnum):
    ZERO_SHOT = "zero_shot"
    ALPACA_SFT = "alpaca_sft"
    QUESTION_ONLY = "question_only"


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text()


def render_zero_shot_prompt(instruction: str) -> str:
    template = load_prompt_template(ZERO_SHOT_SYSTEM_PROMPT_PATH)
    return template.format(instruction=instruction)


def render_alpaca_prompt(instruction: str, response: str = "") -> str:
    template = load_prompt_template(ALPACA_SFT_PROMPT_PATH)
    return template.format(instruction=instruction, response=response)


def render_question_only_prompt(question: str) -> str:
    template = load_prompt_template(QUESTION_ONLY_PROMPT_PATH)
    return template.format(question=question)


def render_prompt(style: PromptStyle | str, instruction: str, response: str = "") -> str:
    prompt_style = PromptStyle(style)
    if prompt_style == PromptStyle.ZERO_SHOT:
        return render_zero_shot_prompt(instruction)
    if prompt_style == PromptStyle.ALPACA_SFT:
        return render_alpaca_prompt(instruction=instruction, response=response)
    if prompt_style == PromptStyle.QUESTION_ONLY:
        if response:
            raise ValueError("question_only prompt style does not support embedding a response")
        return render_question_only_prompt(question=instruction)
    raise ValueError(f"Unsupported prompt style: {style}")
