from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
import torch


DEFAULT_REAL_MODEL = Path("/mnt/FurFagData/Models/Honkware/Qwen3.5-0.8B-exl3-4.0bpw")


def _project_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        source = parent / "experiments" / "single-layer" / "rocm_ext"
        if source.is_dir():
            return parent
    return None


@pytest.fixture(scope="session")
def rocm_runtime():
    hip_version = getattr(torch.version, "hip", None)
    if hip_version is None:
        pytest.skip("requires a ROCm PyTorch build (torch.version.hip is None)")
    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        pytest.skip(f"requires a visible AMD GPU (availability probe failed: {exc})")
    if not available:
        pytest.skip("requires a visible AMD GPU (torch.cuda.is_available() is False)")

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    architecture = str(getattr(properties, "gcnArchName", "")).strip()
    if not architecture:
        pytest.skip("requires an AMD GPU architecture reported through gcnArchName")
    return {
        "hip_version": str(hip_version),
        "architecture": architecture,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
    }


@pytest.fixture(scope="session")
def extension_info(rocm_runtime):
    from unsloth.exllama.rocm_build import load_or_build_extension_info

    info = load_or_build_extension_info()
    assert callable(getattr(info.module, "reconstruct", None))
    assert "exllamav3" not in sys.modules
    return info


@pytest.fixture(scope="session")
def host_oracle(rocm_runtime, tmp_path_factory):
    """Compile the independent portable CPU reconstruction oracle."""

    project_root = _project_root()
    if project_root is None:
        pytest.skip("independent host-oracle sources are absent from this checkout")
    source_dir = project_root / "experiments" / "single-layer" / "rocm_ext"
    source = source_dir / "host_oracle_bindings.cpp"
    if not source.is_file():
        pytest.skip(f"independent host-oracle source is absent: {source}")

    from torch.utils.cpp_extension import load
    from unsloth.exllama.rocm_build import resolve_ninja

    ninja = resolve_ninja()
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    for header in sorted(source_dir.glob("portable_*.h")):
        digest.update(header.name.encode("utf-8"))
        digest.update(header.read_bytes())
    digest.update(str(torch.__version__).encode("utf-8"))
    digest.update(str(getattr(sys.implementation, "cache_tag", "unknown")).encode("utf-8"))
    module_name = f"exl3_h2_host_oracle_{digest.hexdigest()[:16]}"
    build_dir = tmp_path_factory.mktemp("exl3-h2-host-oracle")

    previous_path = os.environ.get("PATH")
    os.environ["PATH"] = os.pathsep.join((str(ninja.parent), previous_path or ""))
    try:
        module = load(
            name=module_name,
            sources=[str(source)],
            extra_include_paths=[str(source_dir)],
            extra_cflags=["-O2"],
            with_cuda=False,
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path

    assert callable(getattr(module, "reconstruct", None))
    return module


@pytest.fixture(scope="session")
def real_model_dir():
    configured = os.environ.get("UNSLOTH_EXL3_TEST_MODEL") or os.environ.get("EXL3_TEST_MODEL")
    path = Path(configured).expanduser() if configured else DEFAULT_REAL_MODEL
    if not path.is_dir():
        pytest.skip(f"real EXL3 checkpoint directory is unavailable: {path}")
    if not (path / "config.json").is_file():
        pytest.skip(f"real EXL3 checkpoint has no config.json: {path}")
    if not any(path.glob("*.safetensors")):
        pytest.skip(f"real EXL3 checkpoint has no safetensors files: {path}")
    return path.resolve()
