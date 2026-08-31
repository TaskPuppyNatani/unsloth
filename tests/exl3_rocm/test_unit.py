"""Tier 1: GPU-free EXL3 ROCm loader and reconstruction-contract tests."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


UNSLOTH_ROOT = Path(__file__).resolve().parents[2]


def _load_source(name: str, relative_path: str):
    path = UNSLOTH_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rocm_hf = _load_source("_test_exl3_rocm_hf", "unsloth/exllama/rocm_hf.py")
rocm_reconstruct = _load_source(
    "_test_exl3_rocm_reconstruct",
    "unsloth/exllama/rocm_reconstruct.py",
)


def _write_safetensors_header(path: Path, tensors: dict[str, dict]) -> Path:
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def _group(prefix: str = "model.layer.proj") -> dict[str, dict]:
    return {
        f"{prefix}.trellis": {"shape": [8, 8, 64], "torch_dtype": torch.int16},
        f"{prefix}.suh": {"shape": [128], "torch_dtype": torch.float16},
        f"{prefix}.svh": {"shape": [128], "torch_dtype": torch.float16},
        f"{prefix}.mul1": {"shape": [], "torch_dtype": torch.int32},
    }


def test_checkpoint_scan_filters_and_maps_exl3_metadata(tmp_path):
    _write_safetensors_header(
        tmp_path / "model.safetensors",
        {
            "__metadata__": {"format": "pt"},
            "model.layer.proj.trellis": {"dtype": "I16", "shape": [8, 8, 64]},
            "model.layer.proj.suh": {"dtype": "F16", "shape": [128]},
            "model.layer.proj.svh": {"dtype": "BF16", "shape": [128]},
            "model.layer.proj.weight": {"dtype": "F16", "shape": [128, 128]},
        },
    )

    index = rocm_hf._scan_checkpoint(str(tmp_path))

    assert set(index) == {
        "model.layer.proj.trellis",
        "model.layer.proj.suh",
        "model.layer.proj.svh",
    }
    assert index["model.layer.proj.trellis"]["torch_dtype"] is torch.int16
    assert index["model.layer.proj.svh"]["torch_dtype"] is torch.bfloat16


def test_checkpoint_scan_rejects_unknown_exl3_dtype(tmp_path):
    _write_safetensors_header(
        tmp_path / "model.safetensors",
        {"model.layer.proj.trellis": {"dtype": "F8_UNKNOWN", "shape": [8, 8, 64]}},
    )

    with pytest.raises(RuntimeError, match="unsupported safetensors dtype"):
        rocm_hf._scan_checkpoint(str(tmp_path))


def test_checkpoint_scan_rejects_truncated_header(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"short")

    with pytest.raises(RuntimeError, match="Invalid safetensors header"):
        rocm_hf._scan_checkpoint(str(tmp_path))


@pytest.mark.parametrize("missing", ("trellis", "suh", "svh"))
def test_group_rejects_incomplete_exl3_bundle(missing):
    prefix = "model.layer.proj"
    index = _group(prefix)
    index.pop(f"{prefix}.{missing}")

    assert rocm_hf._group_for_prefix(index, prefix) == {}


def test_group_accepts_packed_sign_vectors_and_respects_prefix_boundary():
    prefix = "model.layer.proj"
    index = {
        f"{prefix}.trellis": {"shape": [8, 8, 64], "torch_dtype": torch.int16},
        f"{prefix}.su": {"shape": [128], "torch_dtype": torch.int16},
        f"{prefix}.sv": {"shape": [128], "torch_dtype": torch.int16},
        f"{prefix}_other.trellis": {"shape": [8, 8, 64], "torch_dtype": torch.int16},
    }

    group = rocm_hf._group_for_prefix(index, prefix)

    assert set(group) == {f"{prefix}.trellis", f"{prefix}.su", f"{prefix}.sv"}


def test_meta_holder_preserves_shapes_dtypes_and_frozen_weight():
    holder = rocm_hf.RocmExl3HfLinear(128, 128, _group())

    assert holder.key == "model.layer.proj"
    assert holder.trellis.device.type == "meta"
    assert holder.trellis.dtype is torch.int16
    assert tuple(holder.trellis.shape) == (8, 8, 64)
    assert tuple(holder.weight.shape) == (128, 128)
    assert holder.weight.device.type == "meta"
    assert not holder.weight.requires_grad


def test_meta_holder_rejects_empty_group():
    with pytest.raises(RuntimeError, match="empty ROCm EXL3 tensor group"):
        rocm_hf.RocmExl3HfLinear(128, 128, {})


def test_quantizer_discovers_alias_and_replaces_only_complete_linear(tmp_path):
    prefix = "language_model.model.proj"
    raw_group = {
        f"{prefix}.trellis": {"dtype": "I16", "shape": [8, 8, 64]},
        f"{prefix}.suh": {"dtype": "F16", "shape": [128]},
        f"{prefix}.svh": {"dtype": "F16", "shape": [128]},
    }
    _write_safetensors_header(tmp_path / "model.safetensors", raw_group)

    model = torch.nn.Module()
    model.name_or_path = str(tmp_path)
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.proj = torch.nn.Linear(128, 128, bias=False)
    model.model.language_model.unquantized = torch.nn.Linear(4, 4, bias=False)
    quantizer = rocm_hf.RocmExl3HfQuantizer(rocm_hf.RocmExl3Config())

    replacements = quantizer.get_modules_to_replace(model)
    quantizer.replace_modules(model, None, replacements)

    assert set(replacements) == {"model.language_model.proj"}
    assert isinstance(model.model.language_model.proj, rocm_hf.RocmExl3HfLinear)
    assert isinstance(model.model.language_model.unquantized, torch.nn.Linear)


@pytest.mark.parametrize(("hip_version", "gpu_available"), ((None, True), ("7.1", False)))
def test_quantizer_rejects_missing_rocm_requirements(monkeypatch, hip_version, gpu_available):
    quantizer = rocm_hf.RocmExl3HfQuantizer(rocm_hf.RocmExl3Config())
    monkeypatch.setattr(rocm_hf.torch.version, "hip", hip_version)
    monkeypatch.setattr(rocm_hf.torch.cuda, "is_available", lambda: gpu_available)

    with pytest.raises(RuntimeError, match="requires a ROCm-enabled GPU"):
        quantizer.validate_environment()


def test_quantizer_accepts_rocm_gpu_and_preserves_explicit_dtype(monkeypatch):
    quantizer = rocm_hf.RocmExl3HfQuantizer(rocm_hf.RocmExl3Config())
    monkeypatch.setattr(rocm_hf.torch.version, "hip", "7.1")
    monkeypatch.setattr(rocm_hf.torch.cuda, "is_available", lambda: True)

    quantizer.validate_environment()
    assert quantizer.update_torch_dtype(None) is torch.float16
    assert quantizer.update_torch_dtype(torch.bfloat16) is torch.bfloat16


@pytest.mark.parametrize(
    ("inner", "message"),
    (
        (SimpleNamespace(), "no trellis tensor"),
        (SimpleNamespace(trellis=torch.zeros((1, 8, 16), dtype=torch.float32)), "torch.int16"),
        (SimpleNamespace(trellis=torch.zeros((8, 16), dtype=torch.int16)), "3-dimensional"),
        (SimpleNamespace(trellis=torch.zeros((1, 7, 16), dtype=torch.int16)), "128-column"),
        (SimpleNamespace(trellis=torch.zeros((1, 8, 16), dtype=torch.int16), K=0), "K=0"),
        (
            SimpleNamespace(trellis=torch.zeros((1, 8, 16), dtype=torch.int16), K=2),
            "inconsistent packed width",
        ),
        (
            SimpleNamespace(
                trellis=torch.zeros((1, 8, 16), dtype=torch.int16),
                in_features=32,
            ),
            "in_features does not match",
        ),
        (
            SimpleNamespace(
                trellis=torch.zeros((1, 8, 16), dtype=torch.int16),
                out_features=256,
            ),
            "out_features does not match",
        ),
    ),
)
def test_reconstruction_rejects_malformed_groups(monkeypatch, inner, message):
    monkeypatch.setattr(
        rocm_reconstruct,
        "_load_extension",
        lambda: SimpleNamespace(reconstruct=lambda *args: None),
    )

    with pytest.raises(RuntimeError, match=message):
        rocm_reconstruct.reconstruct_inner_weight(inner)


def test_codebook_metadata_flags_and_conflict_detection(monkeypatch):
    assert rocm_reconstruct._inner_flags(SimpleNamespace()) == (False, False)
    assert rocm_reconstruct._inner_flags(SimpleNamespace(mcg=True, mul1=False)) == (
        True,
        False,
    )
    assert rocm_reconstruct._inner_flags(SimpleNamespace(mcg_tensor=object())) == (
        True,
        False,
    )
    assert rocm_reconstruct._inner_flags(SimpleNamespace(mul1_tensor=object())) == (
        False,
        True,
    )

    class _CpuTensorReportingCuda(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda")

    trellis = torch.zeros((1, 8, 16), dtype=torch.int16).as_subclass(_CpuTensorReportingCuda)
    monkeypatch.setattr(
        rocm_reconstruct,
        "_load_extension",
        lambda: SimpleNamespace(reconstruct=lambda *args: None),
    )

    with pytest.raises(RuntimeError, match="cannot use both mcg and mul1"):
        rocm_reconstruct.reconstruct_inner_weight(
            SimpleNamespace(trellis=trellis, mcg=True, mul1=True)
        )


def test_full_weight_contract_transposes_to_out_in_and_casts(monkeypatch):
    inner_weight = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256) / 1024
    holder = SimpleNamespace(suh=torch.ones(128), svh=torch.ones(256))
    monkeypatch.setattr(
        rocm_reconstruct,
        "reconstruct_inner_weight",
        lambda inner: inner_weight.clone(),
    )
    monkeypatch.setattr(
        rocm_reconstruct,
        "_hadamard_128",
        lambda device_type, device_index: torch.eye(128),
    )

    weight = rocm_reconstruct.reconstruct_weight(holder, dtype=torch.bfloat16)

    assert tuple(weight.shape) == (256, 128)
    assert weight.dtype is torch.bfloat16
    expected = inner_weight.to(torch.float16).t().to(torch.bfloat16)
    assert torch.equal(weight, expected)


def test_lightweight_modules_do_not_import_full_exllamav3():
    code = f"""
import importlib.util
import sys
from pathlib import Path

assert "exllamav3" not in sys.modules
for index, relative in enumerate(("unsloth/exllama/rocm_hf.py", "unsloth/exllama/rocm_reconstruct.py")):
    path = Path({str(UNSLOTH_ROOT)!r}) / relative
    spec = importlib.util.spec_from_file_location(f"_h2_lightweight_{{index}}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
assert "exllamav3" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
