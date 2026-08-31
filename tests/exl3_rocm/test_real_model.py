"""Tier 3: opt-in real Qwen3.5 EXL3 ROCm model/training smoke test."""

from __future__ import annotations

import sys

import pytest
import torch


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.rocm,
    pytest.mark.slow,
    pytest.mark.exl3_real_model,
]

_IMMUTABLE_EXL3_SOURCE_TENSORS = (
    "trellis",
    "suh",
    "svh",
    "su",
    "sv",
    "mcg",
    "mcg_tensor",
    "mul1",
    "mul1_tensor",
)


def _exl3_quant_states(model):
    states = []
    for name, module in model.named_modules():
        state = getattr(getattr(module, "weight", None), "quant_state", None)
        if getattr(state, "quant_type", None) == "exl3":
            states.append((name, module, state))
    return states


def _inner_with_trellis(state):
    inner = state.exl3_linear
    while not isinstance(getattr(inner, "trellis", None), torch.Tensor):
        next_inner = getattr(inner, "inner", None)
        if next_inner is None or next_inner is inner:
            raise AssertionError("EXL3 quant state has no trellis tensor")
        inner = next_inner
    return inner


def _snapshot_exl3_source_tensors(inner):
    return {
        name: value.detach().clone()
        for name in _IMMUTABLE_EXL3_SOURCE_TENSORS
        if isinstance((value := getattr(inner, name, None)), torch.Tensor)
    }


def _assert_exl3_source_tensors_unchanged(inner, snapshots, stage):
    for name, before in snapshots.items():
        current = getattr(inner, name, None)
        assert isinstance(current, torch.Tensor), (
            f"EXL3 source tensor {name} disappeared after {stage}"
        )
        assert torch.equal(current, before), f"EXL3 source tensor {name} changed after {stage}"


def test_real_qwen_exl3_load_forward_and_lora_step(
    real_model_dir,
    rocm_runtime,
    extension_info,
):
    assert "exllamav3" not in sys.modules
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(real_model_dir),
        max_seq_length=128,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_exl3=True,
    )

    states = _exl3_quant_states(model)
    assert getattr(model, "_unsloth_exl3_backend", False)
    assert states, "model load attached no EXL3 quant states"
    layer_name, layer, state = states[0]
    reconstructed = state.dequantize(dtype=torch.float16)
    assert tuple(reconstructed.shape) == (state.out_features, state.in_features)
    assert torch.isfinite(reconstructed).all()
    assert not reconstructed.is_inference()

    batch = tokenizer(
        text="ROCm makes local EXL3 LoRA training possible.",
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        forward = model(**batch)
    assert tuple(forward.logits.shape[:2]) == tuple(batch["input_ids"].shape)
    assert torch.isfinite(forward.logits).all()
    assert "exllamav3" not in sys.modules

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        random_state=3407,
    )
    model.train()

    post_lora_states = _exl3_quant_states(model)
    assert post_lora_states
    _, base_layer, state = post_lora_states[0]
    inner = _inner_with_trellis(state)
    source_tensors_before = _snapshot_exl3_source_tensors(inner)
    assert "trellis" in source_tensors_before
    placeholder_before = base_layer.weight.detach().clone()
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all("lora_" in name for name, _ in trainable)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=1e-4)

    batch["labels"] = batch["input_ids"].clone()
    output = model(**batch)
    loss = output.loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    _assert_exl3_source_tensors_unchanged(
        inner,
        source_tensors_before,
        "backward",
    )

    gradients = [
        (name, parameter, parameter.grad)
        for name, parameter in trainable
        if parameter.grad is not None
    ]
    assert len(gradients) == len(trainable)
    assert all(torch.isfinite(gradient).all() for _, _, gradient in gradients)
    nonzero = [
        (name, parameter, gradient)
        for name, parameter, gradient in gradients
        if torch.count_nonzero(gradient).item() > 0
    ]
    minimum_nonzero = (len(trainable) + 1) // 2
    assert len(nonzero) >= minimum_nonzero, (
        "too few trainable LoRA tensors had nonzero first-step gradients: "
        f"{len(nonzero)}/{len(trainable)}; expected at least {minimum_nonzero}"
    )

    updated_name, updated_parameter, _ = nonzero[0]
    updated_before = updated_parameter.detach().clone()
    optimizer.step()
    update_delta = (updated_parameter.detach() - updated_before).float().abs().max().item()

    assert update_delta > 0.0
    _assert_exl3_source_tensors_unchanged(
        inner,
        source_tensors_before,
        "optimizer.step()",
    )
    assert torch.equal(base_layer.weight, placeholder_before)
    assert "exllamav3" not in sys.modules
    print(f"PyTorch: {torch.__version__}; ROCm: {rocm_runtime['hip_version']}")
    print(f"GPU: {rocm_runtime['device_name']}; architecture: {rocm_runtime['architecture']}")
    print(f"Extension: {extension_info.extension_path} ({extension_info.origin})")
    print(f"EXL3 states: {len(states)}; representative layer: {layer_name}")
    print(f"Logits: {tuple(forward.logits.shape)} {forward.logits.dtype}; finite=True")
    print(f"Loss: {float(loss.item())}")
    print(f"LoRA tensors: {len(trainable)}; finite gradients: {len(gradients)}")
    print(f"Nonzero gradients: {len(nonzero)}/{len(trainable)} (minimum {minimum_nonzero})")
    print(f"Updated adapter: {updated_name}; max delta: {update_delta}")
    print("Immutable EXL3 tensors unchanged: " + ", ".join(source_tensors_before))
    print("EXL3 source tensors/placeholder base unchanged: True")
    print("exllamav3 imported: False")
