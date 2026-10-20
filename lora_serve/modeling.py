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
def complete(tok, model, prompts: list[str], max_new_tokens: int = 32) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
    )
    new = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


def timed_complete(tok, model, prompts, max_new_tokens=32):
    t0 = time.perf_counter()
    outs = complete(tok, model, prompts, max_new_tokens)
    return outs, time.perf_counter() - t0


def state_dict_mb(model) -> float:
    """Serialized size of the weights, in MB (what you would ship)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


class Int8WeightLinear(torch.nn.Module):
    """Weight-only int8 linear: int8 weights + one fp32 scale per output row.

    Activations stay fp32; weights are dequantized in forward. Smaller on disk and in memory,
    no activation quantization error.
    """

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale[:, None]).to(torch.int8))
        self.register_buffer("scale", scale)
        self.bias = None if linear.bias is None else torch.nn.Parameter(linear.bias.detach())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.qweight.to(x.dtype) * self.scale[:, None], self.bias)


def quantize_int8(model, skip_lm_head: bool = True):
    """Weight-only int8 quantization of every nn.Linear except (by default) the output head.

    The head is tied to the input embeddings in SmolLM2, so it is left in fp32.
    """
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, torch.nn.Linear) and not (skip_lm_head and full.endswith("lm_head")):
                setattr(mod, child_name, Int8WeightLinear(child))
    return model


def quantize_dynamic_int8(model):
    """PyTorch dynamic int8 (weights and activations) on nn.Linear except the head.

    Kept for comparison: on this 135M model it destroys accuracy (see results).
    """
    names = {n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head")}
    return torch.ao.quantization.quantize_dynamic(model, names, dtype=torch.qint8)
