"""Supervised fine-tuning with LoRA adapters (peft), plain PyTorch training loop.

Loss is only computed on the answer tokens; prompt tokens are masked with -100.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .data import load_splits
from .modeling import BASE_MODEL, load, prompt_text
from .task import answer_json


def build_example(tok, text: str, label: str, max_len: int = 256):
    prompt = prompt_text(tok, text)
    answer = answer_json(label) + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"][: max_len - 16]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


def collate(batch, pad_id):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    return ids, att, lab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default="artifacts/adapter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import mlflow
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok, model = load(args.model)
    tok.padding_side = "right"
    cfg = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    train, _ = load_splits(n_train=args.n_train)
    examples = [build_example(tok, r["text"], r["label"]) for r in train]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = args.epochs * ((len(examples) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("lora-serve")
    with mlflow.start_run(run_name="lora-sft"):
        mlflow.log_params({**vars(args), "trainable_params": trainable, "total_params": total})
        model.train()
        step, t0 = 0, time.time()
        for _ in range(args.epochs):
            random.shuffle(examples)
            for i in range(0, len(examples), args.batch_size):
                ids, att, lab = collate(examples[i : i + args.batch_size], tok.pad_token_id)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                mlflow.log_metric("train_loss", loss.item(), step=step)
                if step % 10 == 0:
                    print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
        train_sec = time.time() - t0
        mlflow.log_metric("train_seconds", train_sec)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        info = {"trainable_params": trainable, "total_params": total, "train_seconds": train_sec,
                "steps": steps, "final_loss": loss.item()}
        Path(args.out, "train_info.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info))


if __name__ == "__main__":
    main()
