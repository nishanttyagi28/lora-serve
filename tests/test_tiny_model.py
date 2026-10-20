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


def test_int8_quantization_shrinks_and_runs():
    from lora_serve.modeling import quantize_int8, state_dict_mb

    m = tiny().eval()
    q = quantize_int8(m)
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        assert q(input_ids=ids).logits.shape == (1, 8, 64)
    assert any(isinstance(x, torch.ao.nn.quantized.dynamic.Linear) for x in q.modules())
    assert state_dict_mb(q) < state_dict_mb(m)
