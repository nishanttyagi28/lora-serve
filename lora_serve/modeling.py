"""Model loading, prompting and greedy decoding helpers (PyTorch + transformers)."""

from __future__ import annotations

import io
import time

import torch

from .task import build_messages

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load(model_id: str = BASE_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return tok, model


def prompt_text(tok, text: str, few_shot: bool = False) -> str:
    return tok.apply_chat_template(
        build_messages(text, few_shot=few_shot), tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def complete(tok, model, prompts: list[str], max_new_tokens: int = 12) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
    )
    new = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


def timed_complete(tok, model, prompts, max_new_tokens=12):
    t0 = time.perf_counter()
    outs = complete(tok, model, prompts, max_new_tokens)
    return outs, time.perf_counter() - t0


def state_dict_mb(model) -> float:
    """Serialized size of the weights, in MB (what you would ship)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def quantize_int8(model):
    """Post-training dynamic int8 quantization of every nn.Linear (CPU inference)."""
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
