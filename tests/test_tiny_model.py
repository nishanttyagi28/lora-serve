"""Offline checks of the training/quantization code paths on a tiny random model."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def tiny():
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg)


def test_collate_masks_prompt():
    from lora_serve.train import collate

    ids, att, lab = collate([([1, 2, 3], [-100, -100, 3]), ([4], [4])], pad_id=0)
    assert ids.shape == (2, 3)
    assert lab[0].tolist() == [-100, -100, 3] and lab[1].tolist() == [4, -100, -100]
    assert att.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_lora_step_only_updates_adapters_and_merges():
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(tiny(), LoraConfig(r=4, target_modules=["q_proj", "v_proj"],
                                              task_type="CAUSAL_LM"))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    ids = torch.randint(0, 64, (2, 8))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(5):
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    merged = model.merge_and_unload()
    assert not any("lora_" in n for n, _ in merged.named_parameters())


def test_int8_weight_only_close_to_fp32_and_smaller():
    from lora_serve.modeling import Int8WeightLinear, quantize_int8, state_dict_mb

    m = tiny().eval()
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        ref = m(input_ids=ids).logits
    size = state_dict_mb(m)
    q = quantize_int8(m)
    with torch.no_grad():
        out = q(input_ids=ids).logits
    assert isinstance(q.model.layers[0].self_attn.q_proj, Int8WeightLinear)
    assert isinstance(q.lm_head, torch.nn.Linear)
    assert torch.allclose(out, ref, atol=0.05)
    assert state_dict_mb(q) < size


def test_int8_weight_rows_scaled():
    from lora_serve.modeling import Int8WeightLinear

    lin = torch.nn.Linear(16, 4)
    q = Int8WeightLinear(lin)
    assert q.qweight.dtype == torch.int8 and q.qweight.abs().max() <= 127
    x = torch.randn(3, 16)
    assert torch.allclose(q(x), lin(x), atol=0.02)


def test_dynamic_int8_runs():
    from lora_serve.modeling import quantize_dynamic_int8

    q = quantize_dynamic_int8(tiny().eval())
    with torch.no_grad():
        assert q(input_ids=torch.randint(0, 64, (1, 8))).logits.shape == (1, 8, 64)
    assert isinstance(q.model.layers[0].mlp.up_proj, torch.ao.nn.quantized.dynamic.Linear)
