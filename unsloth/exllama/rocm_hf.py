# Copyright 2023-present Daniel Han-Chen & the Unsloth team.
#
# Licensed under the Apache License, Version 2.0.
"""Lightweight Transformers EXL3 loader for ROCm.

This module intentionally does not import exllamav3. It only provides the
checkpoint-loading surface needed to materialize EXL3 tensor bundles. After
loading, quant_linear.attach_exl3_quant_states() replaces these holders with
Unsloth's LoRA-facing ExllamaV3Linear wrapper.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers.quantizers.auto import (
    AUTO_QUANTIZATION_CONFIG_MAPPING,
    AUTO_QUANTIZER_MAPPING,
    HfQuantizer,
)
from transformers.quantizers.base import QuantizationConfigMixin
from transformers.utils.quantization_config import QuantizationMethod


_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

if hasattr(torch, "uint16"):
    _DTYPE_MAP["U16"] = torch.uint16
if hasattr(torch, "uint32"):
    _DTYPE_MAP["U32"] = torch.uint32
if hasattr(torch, "uint64"):
    _DTYPE_MAP["U64"] = torch.uint64


_EXL3_SUFFIXES = (
    "trellis",
    "suh",
    "svh",
    "su",
    "sv",
    "mcg",
    "mul1",
    "bias",
)


def _read_safetensors_header(filename: Path) -> dict:
    with filename.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"Invalid safetensors header: {filename}")
        header_len = struct.unpack("<Q", raw)[0]
        header = f.read(header_len)

    data = json.loads(header.decode("utf-8"))
    data.pop("__metadata__", None)
    return data


def _scan_checkpoint(path: str) -> dict[str, dict]:
    root = Path(path)
    tensors: dict[str, dict] = {}

    for filename in sorted(root.glob("*.safetensors")):
        header = _read_safetensors_header(filename)

        for key, meta in header.items():
            suffix = key.rsplit(".", 1)[-1]
            if suffix not in _EXL3_SUFFIXES:
                continue

            dtype_name = str(meta["dtype"])
            dtype = _DTYPE_MAP.get(dtype_name)
            if dtype is None:
                raise RuntimeError(
                    f"Unsloth: unsupported safetensors dtype "
                    f"{dtype_name!r} for EXL3 tensor {key!r}."
                )

            tensors[key] = {
                "shape": list(meta["shape"]),
                "torch_dtype": dtype,
            }

    return tensors


def _group_for_prefix(index: dict[str, dict], prefix: str) -> dict[str, dict]:
    group = {
        key: meta
        for key, meta in index.items()
        if key.rsplit(".", 1)[0] == prefix and key.rsplit(".", 1)[-1] in _EXL3_SUFFIXES
    }

    if f"{prefix}.trellis" not in group:
        return {}

    if f"{prefix}.suh" not in group and f"{prefix}.su" not in group:
        return {}

    if f"{prefix}.svh" not in group and f"{prefix}.sv" not in group:
        return {}

    return group


def _validate_checkpoint_index(index: dict[str, dict]) -> set[str]:
    # A standalone bias also belongs to ordinary dense/norm layers and is not
    # evidence of EXL3. Include it in groups only when EXL3-specific metadata
    # establishes the prefix; all other recognized suffixes establish groups.
    prefixes = {
        key.rsplit(".", 1)[0]
        for key in index
        if key.rsplit(".", 1)[-1] in _EXL3_SUFFIXES and key.rsplit(".", 1)[-1] != "bias"
    }
    if not prefixes:
        raise RuntimeError(
            "Unsloth: the ROCm EXL3 loader found no pre-quantized EXL3 "
            "trellis tensors in the checkpoint. ROCm currently requires a "
            "pre-quantized dense EXL3 checkpoint; dense-to-EXL3 conversion "
            "is not supported."
        )

    for prefix in sorted(prefixes):
        missing = []
        if f"{prefix}.trellis" not in index:
            missing.append("trellis")
        if f"{prefix}.suh" not in index and f"{prefix}.su" not in index:
            missing.append("suh or su")
        if f"{prefix}.svh" not in index and f"{prefix}.sv" not in index:
            missing.append("svh or sv")
        if missing:
            raise RuntimeError(
                f"Unsloth: malformed EXL3 tensor group {prefix!r}; missing {', '.join(missing)}."
            )
        if f"{prefix}.mcg" in index and f"{prefix}.mul1" in index:
            raise RuntimeError(
                f"Unsloth: malformed EXL3 tensor group {prefix!r}; mcg and "
                "mul1 codebooks are mutually exclusive."
            )
    return prefixes


def _ignore_expected_dense_weight_diagnostics(model, module_names) -> None:
    """Filter only placeholder weights for proven EXL3 replacements."""

    patterns = list(getattr(model, "_keys_to_ignore_on_load_missing", None) or ())
    for name in sorted(module_names):
        pattern = rf"^{re.escape(name)}\.weight$"
        if pattern not in patterns:
            patterns.append(pattern)
    model._keys_to_ignore_on_load_missing = patterns


def _meta_tensor(meta: Optional[dict]):
    if meta is None:
        return None

    return torch.empty(
        tuple(meta["shape"]),
        dtype=meta["torch_dtype"],
        device="meta",
    )


class RocmExl3HfLinear(torch.nn.Module):
    """Meta-light EXL3 tensor holder used only during Transformers loading."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        exl3_tensors: dict[str, dict],
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        keys = list(exl3_tensors)
        if not keys:
            raise RuntimeError("Unsloth: empty ROCm EXL3 tensor group.")

        self.key = keys[0].rsplit(".", 1)[0]

        def meta(name):
            return exl3_tensors.get(f"{self.key}.{name}")

        self.register_buffer("trellis", _meta_tensor(meta("trellis")))
        self.register_buffer("suh", _meta_tensor(meta("suh")))
        self.register_buffer("svh", _meta_tensor(meta("svh")))
        self.register_buffer("su", _meta_tensor(meta("su")))
        self.register_buffer("sv", _meta_tensor(meta("sv")))
        self.register_buffer("mcg", _meta_tensor(meta("mcg")))
        self.register_buffer("mul1", _meta_tensor(meta("mul1")))
        self.register_buffer("bias", _meta_tensor(meta("bias")))

        # Transformers 5.x tied-weight handling expects weight to be a Parameter.
        # It remains meta-only and is discarded when attach_exl3_quant_states()
        # installs the LoRA-facing wrapper.
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.out_features, self.in_features),
                dtype=torch.float16,
                device="meta",
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This normally disappears before the first forward, but keeping the
        # holder functional makes loading/finalization failures easier to debug.
        from .rocm_reconstruct import reconstruct_weight

        weight = reconstruct_weight(self, dtype=x.dtype).to(x.device)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)


@dataclass
class RocmExl3Config(QuantizationConfigMixin):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.quant_method = "exl3"


class RocmExl3HfQuantizer(HfQuantizer):
    """Transformers quantizer facade for pre-quantized EXL3 on ROCm."""

    requires_calibration = False
    required_packages = []
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError(
                "Unsloth: ROCm EXL3 loader requires a ROCm-enabled GPU: use "
                "a ROCm-enabled PyTorch build and ensure the AMD GPU is "
                "visible. Verify torch.version.hip and "
                "torch.cuda.is_available()."
            )

    def update_torch_dtype(self, torch_dtype):
        return torch.float16 if torch_dtype is None else torch_dtype

    def get_modules_to_replace(self, model):
        path = getattr(model, "name_or_path", None)

        if not path or not os.path.isdir(path):
            raise ValueError("Unsloth: ROCm EXL3 model must be initialized from a local directory.")

        index = _scan_checkpoint(path)
        checkpoint_prefixes = _validate_checkpoint_index(index)
        modules_to_replace = {}
        consumed_prefixes = set()

        for name, module in tuple(model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue

            candidates = [name]

            if name.startswith("model.language_model."):
                candidates.append("language_model.model." + name[len("model.language_model.") :])

            if name.startswith("language_model.model."):
                candidates.append("model.language_model." + name[len("language_model.model.") :])

            matched_group = None

            for candidate in candidates:
                group = _group_for_prefix(index, candidate)
                if group:
                    matched_group = group
                    consumed_prefixes.add(candidate)
                    break

            if matched_group:
                modules_to_replace[name] = RocmExl3HfLinear(
                    module.in_features,
                    module.out_features,
                    matched_group,
                )

        if not modules_to_replace:
            raise RuntimeError(
                "Unsloth: EXL3 tensors were found, but none matched the "
                "model's dense Linear modules. The checkpoint architecture "
                "or tensor names are not supported by the lightweight ROCm "
                "EXL3 loader."
            )

        # Some model classes intentionally omit auxiliary checkpoint heads
        # (for example Qwen3.5 declares ^mtp.* unexpected keys as ignorable).
        # Validate these groups above, then account for them only when the
        # model's existing declaration covers every tensor and its dense
        # weight counterpart. Never add a new unexpected-key suppression.
        ignored_patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ()
        ignored_prefixes = {
            prefix
            for prefix in checkpoint_prefixes - consumed_prefixes
            if all(
                any(re.search(pattern, key) for pattern in ignored_patterns)
                for key in (*_group_for_prefix(index, prefix), f"{prefix}.weight")
            )
        }
        unconsumed = checkpoint_prefixes - consumed_prefixes - ignored_prefixes
        if unconsumed:
            raise RuntimeError(
                "Unsloth: EXL3 checkpoint groups did not match any supported "
                f"dense Linear module: {', '.join(sorted(unconsumed))}. "
                "Check that the checkpoint config, architecture and tensor "
                "names describe the same model; this layout is not supported "
                "by the lightweight ROCm EXL3 loader."
            )

        return modules_to_replace

    def replace_modules(self, module, path, modules_to_replace):
        replacements = {}

        for name, child in module.named_children():
            key = f"{path}.{name}" if path else name

            replacement = modules_to_replace.get(key)
            if replacement is not None:
                replacements[name] = replacement
            else:
                self.replace_modules(
                    child,
                    key,
                    modules_to_replace,
                )

        for name, replacement in replacements.items():
            setattr(module, name, replacement)

    def _process_model_before_weight_loading(
        self,
        model,
        keep_in_fp32_modules=None,
        **kwargs,
    ):
        modules = self.get_modules_to_replace(model)
        self.replace_modules(model, None, modules)
        installed = dict(model.named_modules())
        if any(installed.get(name) is not replacement for name, replacement in modules.items()):
            raise RuntimeError("Unsloth: ROCm EXL3 module replacement was incomplete.")
        _ignore_expected_dense_weight_diagnostics(model, modules)

        config = kwargs.get("config")
        if config is not None:
            config.tie_word_embeddings = False

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    @property
    def is_trainable(self, model=None):
        return True

    def is_serializable(self, safe_serialization=None):
        return False


def patch_transformers_rocm_exl3() -> bool:
    """Register the lightweight ROCm EXL3 loader with Transformers."""

    AUTO_QUANTIZER_MAPPING["exl3"] = RocmExl3HfQuantizer
    AUTO_QUANTIZATION_CONFIG_MAPPING["exl3"] = RocmExl3Config

    try:
        QuantizationMethod.EXL3 = "exl3"
    except Exception:
        pass

    return True
