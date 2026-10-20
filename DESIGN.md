# Design

## Goal

Show the full loop for a small task-specific model: pick a task with an exact answer, measure
prompting first, fine-tune only if it pays off, shrink the model, and serve it with the usual
production pieces (validation, metrics, container, CI).

## Task

AG News topic classification (4 classes), answered as `{"label": "<class>"}`. A JSON answer
was chosen over a bare label so the same schema check is used in evaluation and in serving.
Scoring is exact match on the parsed label; invalid JSON counts as wrong.

## Model and training

- Base: `HuggingFaceTB/SmolLM2-135M-Instruct`, small enough to fine-tune on CPU.
- LoRA (peft) r=16, alpha=32 on all attention and MLP projections, about 3.5% of parameters (4.9M of 139M)
  trainable. A hand-written PyTorch loop masks prompt tokens so loss covers only the answer.
- 1,200 training examples, 1 epoch, batch 8, AdamW lr 2e-4 with linear decay.
- QLoRA proper (4-bit base via bitsandbytes) needs a CUDA GPU, so it is not used here. The
  quantization part is post-training dynamic int8 on the merged model instead.

## Comparison

All variants run on the same 300 seeded test articles with greedy decoding:
base zero-shot, base few-shot (4 examples), LoRA adapter, merged fp32, merged + dynamic int8.
Latency is measured at batch size 1 on the first 40 articles; size is the serialized
state dict.

## Fine-tuning vs RAG vs prompting

- Prompting is the baseline and costs nothing to try; the results show how far it gets.
- RAG helps when the answer depends on facts the model does not have. Here the label depends
  on the text in the request itself, so retrieval adds nothing.
- Fine-tuning fits when the task is narrow, the output format must be exact, and labelled
  data exists. It also lets a much smaller model do the job, which is what makes CPU serving
  cheap.

## Serving

FastAPI app with three endpoints. Requests pass input guardrails (length bounds and a list of
prompt-injection regexes) before reaching the model. The output is validated against the
JSON schema; on failure the request is retried once with a few-shot prompt, then fails with
502. Prometheus counters track requests by status, guardrail blocks by reason, retries and
labels, plus a latency histogram. The predictor is injected, so tests use a fake one.

## Tracking

MLflow with a local SQLite store (`mlflow.db`): params, per-step training loss, and one run
per evaluated variant. Browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
