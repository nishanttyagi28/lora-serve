"""Predictors used by the API. The HF predictor is loaded lazily so tests stay offline."""

from __future__ import annotations

import os
from typing import Protocol

from .task import parse_label


class Predictor(Protocol):
    name: str

    def raw(self, text: str, attempt: int) -> str: ...


class LabelError(RuntimeError):
    pass


def predict_with_retry(p: Predictor, text: str, max_attempts: int = 2) -> tuple[str, int]:
    """Call the model, validate the JSON output against the schema, retry if invalid."""
    for attempt in range(max_attempts):
        label = parse_label(p.raw(text, attempt))
        if label is not None:
            return label, attempt + 1
    raise LabelError("model output failed schema validation")


class FakePredictor:
    """Keyword model for tests and smoke runs. Can be told to emit bad JSON first."""

    name = "fake"

    def __init__(self, bad_first: int = 0):
        self.bad_first = bad_first
        self.calls = 0

    def raw(self, text: str, attempt: int) -> str:
        self.calls += 1
        if self.calls <= self.bad_first:
            return "label: Sports"
        t = text.lower()
        if any(w in t for w in ("match", "game", "team", "cup")):
            lab = "Sports"
        elif any(w in t for w in ("stock", "shares", "profit", "market")):
            lab = "Business"
        elif any(w in t for w in ("software", "nasa", "chip", "internet")):
            lab = "Sci/Tech"
        else:
            lab = "World"
        return '{"label": "%s"}' % lab


class HFPredictor:
    """Merged LoRA model (optionally int8) served on CPU."""

    def __init__(self, model_dir: str, int8: bool = True):
        from .modeling import load, quantize_int8

        self.tok, model = load(model_dir)
        self.model = quantize_int8(model) if int8 else model
        self.name = os.path.basename(model_dir.rstrip("/")) + ("-int8" if int8 else "")

    def raw(self, text: str, attempt: int) -> str:
        from .modeling import complete, prompt_text

        # second attempt adds few-shot examples to steer the format
        return complete(self.tok, self.model, [prompt_text(self.tok, text, few_shot=attempt > 0)])[0]


def from_env() -> Predictor:
    kind = os.getenv("PREDICTOR", "fake")
    if kind == "fake":
        return FakePredictor()
    return HFPredictor(os.getenv("MODEL_DIR", "artifacts/merged"), int8=os.getenv("INT8", "1") == "1")
