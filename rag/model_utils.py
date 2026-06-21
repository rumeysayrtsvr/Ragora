"""Shared Ollama prompt and response helpers."""
from __future__ import annotations

import re


NO_THINK_INSTRUCTION = (
    "/no_think\n"
    "Thinking mode kapali. Ic muhakeme, analiz, chain-of-thought veya <think> etiketi yazma. "
    "Sadece kullaniciya gosterilecek nihai cevabi ver."
)


def with_no_think(prompt: str) -> str:
    """Prefix prompts with a non-thinking instruction."""
    return f"{NO_THINK_INSTRUCTION}\n\n{prompt}".strip()


def strip_thinking(text: str) -> str:
    """Remove leaked reasoning blocks from model output."""
    if not text:
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*(?:thinking|dusunme|düşünme)\s*:\s*.*?(?=\n\S|$)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned
