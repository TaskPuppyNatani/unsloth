"""Tier 2: ROCm reconstruction and LoRA integration regressions."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open


pytestmark = [pytest.mark.gpu, pytest.mark.rocm]

MASK32 = 0xFFFFFFFF
REAL_TRELLIS = "model.language_model.layers.0.mlp.gate_proj.trellis"


def _signed16(value: int) -> int:
    value &= 0xFFFF
    return value if value < 0x8000 else value - 0x10000


def _packed_pattern(bits: int, pattern: int) -> torch.Tensor:
    values = []
    state = (0x243F6A88 ^ (bits * 0x10001 + pattern)) & MASK32
    words_per_stream = bits * 8
    for warp in range(8):
        for index in range(words_per_stream):
            if pattern == 0:
                word = 0
            elif pattern == 1:
                word = MASK32
            elif pattern == 2:
                word = 0x01234567 if ((warp + index) & 1) else 0x89ABCDEF
            else:
                state ^= (state << 13) & MASK32
                state &= MASK32
                state ^= state >> 17
                state &= MASK32
                state ^= (state << 5) & MASK32
                state &= MASK32
                word = (state ^ (warp * 0x9E3779B9)) & MASK32
            values.extend((_signed16(word), _signed16(word >> 16)))
    return torch.tensor(values, dtype=torch.int16).reshape(1, 8, 16 * bits).contiguous()


def _codebook_flags(codebook: int) -> tuple[bool, bool]:
    return ((False, False), (True, False), (False, True))[codebook]


def _synthetic_inner(bits: int = 4):
    trellis = _packed_pattern(bits, 3).repeat(8, 2, 1).to("cuda").contiguous()
    return SimpleNamespace(
        trellis=trellis,
        K=bits,
        in_features=128,
        out_features=256,
        suh=torch.ones(128, dtype=torch.float16, device="cuda"),
        svh=torch.ones(256, dtype=torch.float16, device="cuda"),
        mcg=False,
        mul1=True,
    )


def _checkpoint_tensor(model_dir, key: str) -> torch.Tensor:
    for filename in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(filename), framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key).contiguous()
    raise AssertionError(f"checkpoint tensor is missing: {key}")


def test_automatic_extension_load_and_synthetic_oracle_matrix(
    extension_info,
    host_oracle,
    rocm_runtime,
):
    cases = 0
    for bits in range(1, 9):
        for codebook in range(3):
            mcg, mul1 = _codebook_flags(codebook)
            for pattern in range(4):
                packed_cpu = _packed_pattern(bits, pattern)
                expected_bits = host_oracle.reconstruct(
                    packed_cpu,
                    bits,
                    mcg,
                    mul1,
                ).contiguous()
                packed_gpu = packed_cpu.to("cuda")
                output = torch.empty((16, 128), dtype=torch.float16, device="cuda")
                extension_info.module.reconstruct(output, packed_gpu, bits, mcg, mul1)
                actual_bits = output.cpu().contiguous().view(torch.int16)
                assert torch.equal(actual_bits, expected_bits), (
                    f"mismatch for K={bits}, codebook={codebook}, pattern={pattern}"
                )
                cases += 1

    assert cases == 96
    assert "exllamav3" not in sys.modules
    print(
        f"PASS: {cases} synthetic HIP/oracle cases on "
        f"{rocm_runtime['architecture']} ({extension_info.origin})"
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_reconstructed_weight_orientation_dtype_and_linear(extension_info, monkeypatch, dtype):
    from unsloth.exllama import rocm_reconstruct

    monkeypatch.setattr(rocm_reconstruct, "_EXT", extension_info.module)
    inner = _synthetic_inner()
    weight = rocm_reconstruct.reconstruct_weight(inner, dtype=dtype)
    inputs = torch.randn((2, 3, 128), dtype=dtype, device="cuda") * 0.05
    output = F.linear(inputs, weight)

    assert tuple(weight.shape) == (256, 128)
    assert weight.dtype is dtype
    assert tuple(output.shape) == (2, 3, 256)
    assert torch.isfinite(weight).all()
    assert torch.isfinite(output).all()
    assert "exllamav3" not in sys.modules


@pytest.mark.parametrize("mcg,mul1", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize("noncontiguous", (False, True))
def test_inner_reconstruction_dispatches_whole_trellis_once(
    extension_info,
    monkeypatch,
    mcg,
    mul1,
    noncontiguous,
):
    from unsloth.exllama import rocm_reconstruct

    class CountingExtension:
        def __init__(self, module):
            self.module = module
            self.calls = []

        def reconstruct(self, output, packed, bits, use_mcg, use_mul1):
            self.calls.append(
                (
                    tuple(output.shape),
                    tuple(packed.shape),
                    packed.is_contiguous(),
                    bits,
                    use_mcg,
                    use_mul1,
                )
            )
            return self.module.reconstruct(output, packed, bits, use_mcg, use_mul1)

    extension = CountingExtension(extension_info.module)
    monkeypatch.setattr(rocm_reconstruct, "_EXT", extension)
    inner = _synthetic_inner()
    inner.mcg = mcg
    inner.mul1 = mul1
    if noncontiguous:
        inner.trellis = inner.trellis.transpose(0, 1).contiguous().transpose(0, 1)
        assert not inner.trellis.is_contiguous()
    expected = torch.empty((128, 256), dtype=torch.float16, device="cuda")
    extension_info.module.reconstruct(
        expected,
        inner.trellis.contiguous(),
        inner.K,
        mcg,
        mul1,
    )

    actual = rocm_reconstruct.reconstruct_inner_weight(inner)

    assert extension.calls == [((128, 256), (8, 16, 64), True, 4, mcg, mul1)]
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert "exllamav3" not in sys.modules


def test_mutually_exclusive_codebooks_are_rejected(extension_info, monkeypatch):
    from unsloth.exllama import rocm_reconstruct

    monkeypatch.setattr(rocm_reconstruct, "_EXT", extension_info.module)
    inner = _synthetic_inner()
    inner.mcg = True
    inner.mul1 = True

    with pytest.raises(RuntimeError, match="cannot use both mcg and mul1"):
        rocm_reconstruct.reconstruct_inner_weight(inner)


def test_unsloth_lora_matmul_backward_and_frozen_base(extension_info, monkeypatch):
    from unsloth.exllama import rocm_reconstruct
    from unsloth.exllama.quant_linear import Exl3QuantState
    from unsloth.kernels.utils import matmul_lora

    monkeypatch.setattr(rocm_reconstruct, "_EXT", extension_info.module)
    inner = _synthetic_inner()
    trellis_before = inner.trellis.clone()
    quant_state = Exl3QuantState(
        inner,
        in_features=128,
        out_features=256,
        compute_dtype=torch.float16,
    )
    placeholder = torch.zeros((256, 1), dtype=torch.float16, device="cuda")
    placeholder.quant_state = quant_state
    torch.manual_seed(3407)
    inputs = torch.randn((2, 3, 128), dtype=torch.float16, device="cuda") * 0.05
    inputs.requires_grad_(True)
    lora_a = torch.nn.Parameter(torch.randn((8, 128), dtype=torch.float16, device="cuda") * 0.01)
    lora_b = torch.nn.Parameter(torch.randn((256, 8), dtype=torch.float16, device="cuda") * 0.01)

    dense_weight = quant_state.dequantize(dtype=torch.float16)
    assert not dense_weight.is_inference()
    actual = matmul_lora(inputs, placeholder, quant_state, lora_a, lora_b, 2.0)
    with torch.no_grad():
        expected = F.linear(inputs, dense_weight)
        expected += F.linear(F.linear(inputs, lora_a), lora_b) * 2.0
    assert torch.allclose(actual.float(), expected.float(), rtol=5e-3, atol=2e-3)

    actual.float().square().mean().backward()

    for name, tensor in (("input", inputs), ("LoRA A", lora_a), ("LoRA B", lora_b)):
        assert tensor.grad is not None, f"{name} gradient missing"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient non-finite"
        assert tensor.grad.float().norm().item() > 0.0, f"{name} gradient zero"
    assert torch.equal(inner.trellis, trellis_before)
    assert "exllamav3" not in sys.modules


@pytest.mark.exl3_real_model
def test_real_layer_matches_independent_oracle_bit_exact(
    real_model_dir,
    rocm_runtime,
    extension_info,
    host_oracle,
):
    packed_cpu = _checkpoint_tensor(real_model_dir, REAL_TRELLIS)
    mul1 = _checkpoint_tensor(real_model_dir, REAL_TRELLIS.replace(".trellis", ".mul1"))
    assert tuple(packed_cpu.shape) == (64, 224, 64)
    assert packed_cpu.dtype is torch.int16
    assert int(mul1.item()) == -2082680531

    output = torch.empty((1024, 3584), dtype=torch.float16, device="cuda")
    extension_info.module.reconstruct(output, packed_cpu.to("cuda"), 4, False, True)
    actual_bits = output.cpu().contiguous().view(torch.int16)
    checked_blocks = 0
    for row_block in range(64):
        for column_block in range(28):
            packed_block = packed_cpu[
                row_block : row_block + 1,
                column_block * 8 : (column_block + 1) * 8,
                :,
            ].contiguous()
            expected_bits = host_oracle.reconstruct(
                packed_block,
                4,
                False,
                True,
            ).contiguous()
            actual_block = actual_bits[
                row_block * 16 : (row_block + 1) * 16,
                column_block * 128 : (column_block + 1) * 128,
            ].contiguous()
            assert torch.equal(actual_block, expected_bits), (
                f"real checkpoint mismatch at block ({row_block}, {column_block})"
            )
            checked_blocks += 1

    assert checked_blocks == 1792
    assert actual_bits.numel() == 3_670_016
    assert "exllamav3" not in sys.modules
    print("PASS: 1792 blocks and 3670016 FP16 values matched bit-exactly")
