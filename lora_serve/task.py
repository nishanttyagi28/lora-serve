"""Task definition: AG News topic classification with a JSON answer.

The model must answer with exactly {"label": "<one of LABELS>"}. Scoring is exact match
on the parsed label; anything that is not valid JSON matching the schema counts as wrong.
"""

from __future__ import annotations

import json
import re

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": LABELS}},
    "required": ["label"],
    "additionalProperties": False,
}

SYSTEM = (
    "You classify news articles by topic. Reply with JSON only, in the form "
    '{"label": "<topic>"} where <topic> is one of: World, Sports, Business, Sci/Tech.'
)

FEW_SHOT = [
    ("Stocks slid on Wall Street after the Fed hinted at another rate rise.", "Business"),
    ("The Yankees beat the Red Sox 5-3 behind two home runs from Rodriguez.", "Sports"),
    ("NASA said its new probe will map the surface of Mars in high detail.", "Sci/Tech"),
    ("Talks between the two governments broke down over the border dispute.", "World"),
]


def answer_json(label: str) -> str:
    return json.dumps({"label": label})


def build_messages(text: str, few_shot: bool = False) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for ex_text, ex_label in FEW_SHOT:
            msgs.append({"role": "user", "content": f"Article: {ex_text}"})
            msgs.append({"role": "assistant", "content": answer_json(ex_label)})
    msgs.append({"role": "user", "content": f"Article: {text}"})
    return msgs


_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_label(output: str) -> str | None:
    """Return the label if output contains a schema-valid JSON object, else None."""
    from jsonschema import ValidationError, validate

    m = _JSON_RE.search(output or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        validate(obj, OUTPUT_SCHEMA)
    except (json.JSONDecodeError, ValidationError):
        return None
    return obj["label"]
