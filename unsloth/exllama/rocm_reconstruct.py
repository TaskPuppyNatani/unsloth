# Copyright 2023-present Daniel Han-Chen & the Unsloth team.
#
# Licensed under the Apache License, Version 2.0.
"""Correctness-first EXL3 weight reconstruction for ROCm.

This module deliberately has no dependency on exllamav3 itself.  The upstream
ExLlamaV3 package imports its complete CUDA extension at package import time,
which is not currently portable to ROCm.  For LoRA training we only need the
EXL3 weight reconstruction operation.

The HIP reconstruction extension is loaded separately and the remaining
Hadamard/sign-vector transforms are ordinary PyTorch operations.

Input convention:
    EXL3 inner weight: [in_features, out_features]

Return convention:
    PyTorch / Unsloth weight: [out_features, in_features]
"""

from __future__ import annotations

import importlib.util
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


_DEFAULT_BUILD_DIR = Path("/var/tmp/exl3-rocm-extension-build")
_EXT = None


def _extension_candidates() -> list[Path]:
    exact = os.environ.get("UNSLOTH_EXL3_ROCM_EXTENSION")
    if exact:
        return [Path(exact)]

    build_dir = Path(
        os.environ.get(
            "EXL3_ROCM_BUILD_DIR",
            str(_DEFAULT_BUILD_DIR),
        )
    )
    return sorted(build_dir.glob("*.so"))


def _module_name(path: Path) -> str:
    name = path.name

    if ".cpython-" in name:
        return name.split(".cpython-", 1)[0]
    if ".abi3" in name:
        return name.split(".abi3", 1)[0]

    return name.rsplit(".so", 1)[0]


def _load_extension():
    global _EXT

    if _EXT is not None:
        return _EXT

    if torch.version.hip is None:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm reconstruction requested with a non-ROCm "
            "PyTorch build."
        )

    errors: list[str] = []

    for path in _extension_candidates():
        if not path.is_file():
            errors.append(f"{path}: file does not exist")
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                _module_name(path),
                path,
            )
            if spec is None or spec.loader is None:
                errors.append(f"{path}: could not create import spec")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "reconstruct"):
                errors.append(f"{path}: no reconstruct() entry point")
                continue

            _EXT = module
            return module

        except Exception as exc:
            errors.append(f"{path}: {exc}")

    detail = "\n".join(errors) if errors else "no candidate extension found"

    raise RuntimeError(
        "Unsloth: EXL3 ROCm reconstruction extension could not be loaded.\n"
        f"{detail}"
    )


@lru_cache(maxsize=8)
def _hadamard_128(
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)

    h = torch.ones(
        (1, 1),
        dtype=torch.float32,
        device=device,
    )

    while h.shape[0] < 128:
        h = torch.cat(
            (
                torch.cat((h, h), dim=1),
                torch.cat((h, -h), dim=1),
            ),
            dim=0,
        )

    return h * (1.0 / math.sqrt(128.0))


def _inner_flags(inner: Any) -> tuple[bool, bool]:
    mcg = getattr(inner, "mcg", None)
    mul1 = getattr(inner, "mul1", None)

    if mcg is None:
        mcg = getattr(inner, "mcg_tensor", None) is not None

    if mul1 is None:
        mul1 = getattr(inner, "mul1_tensor", None) is not None

    return bool(mcg), bool(mul1)


@torch.no_grad()
def reconstruct_inner_weight(inner: Any) -> torch.Tensor:
    """Reconstruct EXL3 trellis data to FP16 ``[in, out]``."""

    ext = _load_extension()

    trellis = getattr(inner, "trellis", None)

    if not isinstance(trellis, torch.Tensor):
        raise RuntimeError(
            "Unsloth: EXL3 ROCm layer has no trellis tensor."
        )

    if trellis.dtype != torch.int16:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm trellis must be torch.int16, "
            f"got {trellis.dtype}."
        )

    if trellis.ndim != 3:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm trellis must be 3-dimensional, "
            f"got shape {tuple(trellis.shape)}."
        )

    if trellis.shape[1] % 8 != 0:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm trellis output dimension is not "
            "128-column aligned."
        )

    K = int(
        getattr(
            inner,
            "K",
            trellis.shape[-1] // 16,
        )
    )

    if not 1 <= K <= 8:
        raise RuntimeError(
            f"Unsloth: unsupported EXL3 trellis K={K}; expected 1..8."
        )

    expected_last = 16 * K

    if trellis.shape[-1] != expected_last:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm trellis has inconsistent packed width: "
            f"{trellis.shape[-1]} != {expected_last}."
        )

    in_features = int(
        getattr(
            inner,
            "in_features",
            trellis.shape[0] * 16,
        )
    )
    out_features = int(
        getattr(
            inner,
            "out_features",
            trellis.shape[1] * 16,
        )
    )

    expected_in = trellis.shape[0] * 16
    expected_out = trellis.shape[1] * 16

    if in_features != expected_in:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm in_features does not match trellis: "
            f"{in_features} != {expected_in}."
        )

    if out_features != expected_out:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm out_features does not match trellis: "
            f"{out_features} != {expected_out}."
        )

    device = trellis.device

    if device.type != "cuda":
        raise RuntimeError(
            "Unsloth: EXL3 ROCm trellis must reside on the GPU, "
            f"got {device}."
        )

    mcg, mul1 = _inner_flags(inner)

    if mcg and mul1:
        raise RuntimeError(
            "Unsloth: EXL3 layer cannot use both mcg and mul1 codebooks."
        )

    weight = torch.empty(
        (in_features, out_features),
        dtype=torch.float16,
        device=device,
    )

    row_blocks = trellis.shape[0]
    col_blocks = trellis.shape[1] // 8

    for row_block in range(row_blocks):
        row_start = row_block * 16

        for col_block in range(col_blocks):
            col_start = col_block * 128

            packed = trellis[
                row_block : row_block + 1,
                col_block * 8 : (col_block + 1) * 8,
                :,
            ].contiguous()

            tile = torch.empty(
                (16, 128),
                dtype=torch.float16,
                device=device,
            )

            ext.reconstruct(
                tile,
                packed,
                K,
                mcg,
                mul1,
            )

            weight[
                row_start : row_start + 16,
                col_start : col_start + 128,
            ] = tile

    return weight


@torch.no_grad()
def reconstruct_weight(
    inner: Any,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a dense EXL3 weight in PyTorch ``[out, in]`` convention."""

    weight = reconstruct_inner_weight(inner)

    k, n = weight.shape

    if k % 128 != 0 or n % 128 != 0:
        raise RuntimeError(
            "Unsloth: EXL3 ROCm Hadamard transform requires dimensions "
            f"divisible by 128, got {(k, n)}."
        )

    suh = getattr(inner, "suh", None)
    svh = getattr(inner, "svh", None)

    if not isinstance(suh, torch.Tensor):
        raise RuntimeError(
            "Unsloth: EXL3 ROCm layer has no unpacked suh tensor."
        )

    if not isinstance(svh, torch.Tensor):
        raise RuntimeError(
            "Unsloth: EXL3 ROCm layer has no unpacked svh tensor."
        )

    if tuple(suh.shape) != (k,):
        raise RuntimeError(
            f"Unsloth: EXL3 suh shape {tuple(suh.shape)} != {(k,)}."
        )

    if tuple(svh.shape) != (n,):
        raise RuntimeError(
            f"Unsloth: EXL3 svh shape {tuple(svh.shape)} != {(n,)}."
        )

    device = weight.device
    h128 = _hadamard_128(device.type, device.index)

    suh = suh.to(
        device=device,
        dtype=torch.float16,
    )
    svh = svh.to(
        device=device,
        dtype=torch.float16,
    )

    # Mirror ExLlamaV3 preapply_had_l:
    # float32 matrix multiplication followed by cast back to fp16.
    weight = weight.float()
    weight = (
        h128 @ weight.view(-1, 128, n)
    ).view(k, n)
    weight = weight.to(torch.float16)

    weight *= suh.unsqueeze(1)

    # Mirror ExLlamaV3 preapply_had_r.
    weight = weight.float()
    weight = (
        weight.view(k, -1, 128) @ h128
    ).view(k, n)
    weight = weight.to(torch.float16)

    weight *= svh.unsqueeze(0)

    if not torch.isfinite(weight).all():
        raise RuntimeError(
            "Unsloth: EXL3 ROCm reconstruction produced non-finite values."
        )

    # ExLlamaV3 returns [in, out].
    # Unsloth / torch.nn.functional.linear expect [out, in].
    weight = weight.t().contiguous()

    if dtype is not None and weight.dtype != dtype:
        weight = weight.to(dtype)

    return weight
