"""Input guardrails (length, prompt-injection patterns) and output schema validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 2000
MIN_CHARS = 3

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|system) ",
    r"forget (everything|all|your) (above|instructions|rules)",
    r"\byou are now\b",
    r"\bact as\b",
    r"(reveal|print|show|repeat) (me )?(the |your )?(system prompt|instructions|hidden prompt)",
    r"<\|?(system|im_start|im_end)\|?>",
    r"\bjailbreak\b",
    r"\bDAN mode\b",
    r"new instructions?:",
    r"respond (only )?with .*label.*(instead|regardless)",
]
_INJ = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


def check_input(text: str) -> Verdict:
    if not isinstance(text, str):
        return Verdict(False, "text must be a string")
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return Verdict(False, "text too short")
    if len(text) > MAX_CHARS:
        return Verdict(False, f"text longer than {MAX_CHARS} characters")
    for pat in _INJ:
        if pat.search(text):
            return Verdict(False, "possible prompt injection")
    return Verdict(True)
