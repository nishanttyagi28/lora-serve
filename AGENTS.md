# AGENTS.md

Notes for anyone (person or tool) changing this repo.

## Commands

- install: `make install` (CPU torch wheel, then `pip install -e ".[ml,dev]"`)
- lint: `ruff check .`
- tests: `pytest -q` (offline, under 15 s, no model download)
- train LoRA adapter: `python -m lora_serve.train` (~8 min on 8 CPU cores)
- compare variants: `python -m lora_serve.evaluate` then `python scripts/plot_results.py`
- merge for serving: `python -m lora_serve.export`
- serve: `PREDICTOR=hf uvicorn lora_serve.app:app` (or `PREDICTOR=fake` for a keyword stub)

## Tests

- Unit tests must stay offline and fast. Use `FakePredictor` or the tiny random Llama in
  `tests/test_tiny_model.py`; never download a real model in a test.
- Any change to guardrails needs a test for the new pattern and one for a normal article that
  must still pass.

## PR rules

- Small focused commits, plain lowercase messages.
- CI (lint + tests) must be green before merge.
- If a change affects accuracy, latency or size, rerun `evaluate` and commit the new
  `results/results.json` and chart in the same PR. Never edit numbers by hand.

## Always

- Keep `results/` numbers exactly as written by `lora_serve.evaluate`.
- Keep the output schema in `lora_serve/task.py` as the single source of truth.
- Expose new service behaviour as a Prometheus metric when it can fail.

## Ask first

- Changing the base model, dataset, or test split size/seed (results stop being comparable).
- Adding a new runtime dependency or changing the Docker base image.
- Loosening a guardrail.

## Never

- Commit model weights, adapters, `mlflow.db` or `mlruns/`.
- Commit secrets or tokens.
- Report a metric that was not produced by a real run.
