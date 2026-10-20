"""Load AG News from the Hugging Face hub and make small, seeded train/test splits."""

from __future__ import annotations

from .task import LABELS

DATASET = "fancyzhx/ag_news"


def load_splits(n_train: int = 1200, n_test: int = 300, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset(DATASET)
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    test = ds["test"].shuffle(seed=seed).select(range(n_test))

    def conv(split):
        return [{"text": r["text"].strip(), "label": LABELS[r["label"]]} for r in split]

    return conv(train), conv(test)
