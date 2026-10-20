"""Compare base zero-shot, base few-shot, LoRA and LoRA merged + int8 on the same test split."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from .data import load_splits
from .modeling import BASE_MODEL, complete, load, prompt_text, quantize_int8, state_dict_mb
from .task import parse_label


def score(tok, model, rows, few_shot, batch_size, n_latency):
    preds, valid = [], 0
    t0 = time.perf_counter()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        outs = complete(tok, model, [prompt_text(tok, r["text"], few_shot) for r in chunk])
        for o in outs:
            lab = parse_label(o)
            valid += lab is not None
            preds.append(lab)
    total = time.perf_counter() - t0
    lat = []
    for r in rows[:n_latency]:
        s = time.perf_counter()
        complete(tok, model, [prompt_text(tok, r["text"], few_shot)])
        lat.append((time.perf_counter() - s) * 1000)
    correct = sum(p == r["label"] for p, r in zip(preds, rows, strict=True))
    lat.sort()
    return {
        "accuracy": correct / len(rows),
        "json_valid_rate": valid / len(rows),
        "n": len(rows),
        "latency_ms_p50": statistics.median(lat),
        "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))],
        "batch_eval_seconds": total,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter", default="artifacts/adapter")
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-latency", type=int, default=40)
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args(argv)

    import mlflow
    from peft import PeftModel

    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    _, test = load_splits(n_test=args.n_test)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("lora-serve")
    results = {"model": args.model, "dataset": "fancyzhx/ag_news", "n_test": len(test),
               "threads": torch.get_num_threads(), "variants": {}}

    def run(name, tok, model, few_shot=False):
        print(f"evaluating {name}", flush=True)
        res = score(tok, model, test, few_shot, args.batch_size, args.n_latency)
        res["weights_mb"] = state_dict_mb(model)
        results["variants"][name] = res
        with mlflow.start_run(run_name=f"eval-{name}"):
            mlflow.log_params({"variant": name, "model": args.model, "n_test": len(test)})
            mlflow.log_metrics(res)
        print(name, json.dumps(res), flush=True)

    tok, base = load(args.model)
    run("base_zero_shot", tok, base)
    run("base_few_shot", tok, base, few_shot=True)

    lora = PeftModel.from_pretrained(base, args.adapter)
    lora.eval()
    run("lora", tok, lora)

    merged = lora.merge_and_unload()
    merged.eval()
    run("lora_merged_fp32", tok, merged)
    run("lora_merged_int8", tok, quantize_int8(merged))

    info = Path(args.adapter, "train_info.json")
    if info.exists():
        results["train"] = json.loads(info.read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
