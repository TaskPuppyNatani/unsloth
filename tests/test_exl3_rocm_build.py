from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "exllama" / "rocm_build.py"
SPEC = importlib.util.spec_from_file_location("_test_exl3_rocm_build", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
rocm_build = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rocm_build
SPEC.loader.exec_module(rocm_build)


def _context(**changes):
    base = rocm_build.BuildContext(
        schema=2,
        torch_version="2.10.0+rocm7.1",
        hip_version="7.1.25424",
        python_cache_tag="cpython-313",
        python_soabi="cpython-313-x86_64-linux-gnu",
        cxx11_abi="True",
        machine="x86_64",
        architectures=("gfx1100",),
        rocm_home="/opt/rocm-test",
        hipcc="/opt/rocm-test/bin/hipcc",
        source_fingerprint="a" * 64,
    )
    return replace(base, **changes)


def _source_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    for index, name in enumerate(rocm_build._SOURCE_FILES):
        (root / name).write_text(f"source-{index}\n", encoding="utf-8")
    return root


def _load_build_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_non_rocm_pytorch_is_rejected(monkeypatch):
    monkeypatch.setattr(rocm_build.torch.version, "hip", None)

    with pytest.raises(rocm_build.RocmExtensionError, match="torch.version.hip is None"):
        rocm_build.load_or_build_extension()


def test_explicit_architectures_are_normalized(monkeypatch):
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100; gfx942,gfx1100")

    assert rocm_build.detect_architectures() == ("gfx1100", "gfx942")


def test_invalid_architecture_is_rejected(monkeypatch):
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100 --unexpected-flag")

    with pytest.raises(rocm_build.RocmExtensionError, match="invalid PYTORCH_ROCM_ARCH"):
        rocm_build.detect_architectures()


def test_active_amd_architecture_is_detected(monkeypatch):
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.setattr(rocm_build.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rocm_build.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        rocm_build.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(gcnArchName="gfx1151" if index == 2 else ""),
    )

    assert rocm_build.detect_architectures() == ("gfx1151",)


def test_source_fingerprint_changes_with_source(tmp_path):
    source_dir = _source_tree(tmp_path / "source")
    before = rocm_build.source_fingerprint(source_dir)
    (source_dir / "reconstruct_rocm.cu").write_text("changed\n", encoding="utf-8")

    assert rocm_build.source_fingerprint(source_dir) != before


def test_packaged_sources_are_the_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_SOURCE_DIR", raising=False)

    source_dir = rocm_build.find_source_dir()

    assert source_dir == MODULE_PATH.parent / "rocm_ext"
    assert {path.name for path in source_dir.iterdir() if path.is_file()} == set(
        rocm_build._SOURCE_FILES
    ) | {"LICENSE.exllamav3", "NOTICE.md"}
    assert "experiments" not in source_dir.parts


def test_source_fingerprint_changes_when_packaged_source_changes(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_SOURCE_DIR", raising=False)
    staged = tmp_path / "installed" / "unsloth" / "exllama" / "rocm_ext"
    shutil.copytree(rocm_build.find_source_dir(), staged)
    before = rocm_build.source_fingerprint(staged)
    source = staged / "portable_bitops.h"
    source.write_text(source.read_text(encoding="utf-8") + "// changed\n", encoding="utf-8")

    assert rocm_build.source_fingerprint(staged) != before


def test_explicit_source_override_wins(monkeypatch, tmp_path):
    override = _source_tree(tmp_path / "override")
    monkeypatch.setenv("UNSLOTH_EXL3_ROCM_SOURCE_DIR", str(override))

    assert rocm_build.find_source_dir() == override


def test_missing_packaged_sources_error_is_actionable(monkeypatch, tmp_path):
    missing = tmp_path / "installed" / "unsloth" / "exllama" / "rocm_ext"
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_SOURCE_DIR", raising=False)
    monkeypatch.setattr(rocm_build, "_packaged_source_dir", lambda: missing)

    with pytest.raises(rocm_build.RocmExtensionError, match="packaged.*incomplete") as error:
        rocm_build.find_source_dir()

    message = str(error.value)
    assert "Reinstall Unsloth" in message
    assert "UNSLOTH_EXL3_ROCM_SOURCE_DIR" in message


def test_package_relative_lookup_works_from_staged_tree(monkeypatch, tmp_path):
    package_dir = tmp_path / "wheel-root" / "unsloth" / "exllama"
    native_dir = package_dir / "rocm_ext"
    package_dir.mkdir(parents=True)
    shutil.copyfile(MODULE_PATH, package_dir / MODULE_PATH.name)
    _source_tree(native_dir)
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_SOURCE_DIR", raising=False)

    staged = _load_build_module(
        package_dir / MODULE_PATH.name,
        "_test_staged_exl3_rocm_build",
    )

    assert staged.find_source_dir() == native_dir


def test_package_data_declares_every_native_source_and_notice():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 test jobs.
        from setuptools._vendor import tomli as tomllib

    pyproject = MODULE_PATH.parents[2] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    declared = set(data["tool"]["setuptools"]["package-data"]["unsloth.exllama"])

    notices = {"LICENSE.exllamav3", "NOTICE.md"}
    assert declared == {f"rocm_ext/{name}" for name in (*rocm_build._SOURCE_FILES, *notices)}
    source_dir = MODULE_PATH.parent / "rocm_ext"
    license_text = (source_dir / "LICENSE.exllamav3").read_text(encoding="utf-8")
    assert "Copyright (c) 2025 Turboderp" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert "copies or substantial portions of the Software" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text
    assert "0c49587a7c235e6303a6bbedc8b665272ad3a2ea" in (source_dir / "NOTICE.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("torch_version", "2.11.0+rocm7.2"),
        ("hip_version", "7.2.0"),
        ("python_soabi", "cpython-314-x86_64-linux-gnu"),
        ("cxx11_abi", "False"),
        ("architectures", ("gfx942",)),
        ("rocm_home", "/opt/rocm-other"),
        ("hipcc", "/opt/rocm-other/bin/hipcc"),
        ("source_fingerprint", "b" * 64),
    ),
)
def test_cache_key_covers_compatibility_axes(field, value):
    context = _context()

    assert replace(context, **{field: value}).key != context.key


def test_cache_root_uses_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert rocm_build.cache_root() == tmp_path / "unsloth" / "exl3-rocm"


def test_missing_toolchain_error_is_actionable(monkeypatch, tmp_path):
    missing = tmp_path / "missing-rocm"
    context = _context(
        rocm_home=str(missing),
        hipcc=str(missing / "bin" / "hipcc"),
    )

    with pytest.raises(rocm_build.RocmExtensionError, match="usable HIP compiler") as error:
        rocm_build.resolve_build_tools(context)

    assert str(missing) in str(error.value)


@pytest.mark.parametrize("already_imported", (False, True))
@pytest.mark.parametrize("fail_inside", (False, True))
def test_nonstandard_override_selects_real_pytorch_hip_linkage(already_imported, fail_inside):
    # A subprocess prevents the test harness's own cpp_extension imports from
    # making the fresh-import case accidentally exercise an already loaded one.
    code = f"""
import importlib.util
import inspect
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch
import torch

torch.version.hip = "7.1.0"
os.environ.pop("ROCM_HOME", None)
os.environ.pop("ROCM_PATH", None)
os.environ["UNSLOTH_EXL3_ROCM_HOME"] = "/nonstandard/review-rocm"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
os.environ.pop("UNSLOTH_EXL3_ROCM_SOURCE_DIR", None)
spec = importlib.util.spec_from_file_location("_override_build", {str(MODULE_PATH)!r})
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)
original_which = shutil.which
original_exists = os.path.exists
def which(name, *args, **kwargs):
    return None if name == "hipcc" else original_which(name, *args, **kwargs)
def exists(path):
    return False if str(path) == "/opt/rocm" else original_exists(path)

with patch.object(shutil, "which", which), patch.object(os.path, "exists", exists):
    assert "torch.utils.cpp_extension" not in sys.modules
    if {already_imported!r}:
        from torch.utils import cpp_extension
        assert cpp_extension.ROCM_HOME is None
        assert cpp_extension.IS_HIP_EXTENSION is False
    context, _ = builder.make_build_context()
    assert context.rocm_home == "/nonstandard/review-rocm"
    tools = builder.BuildTools(Path(context.rocm_home), Path(context.hipcc), "test", Path("/test/ninja"))
    before_env = os.environ.copy()
    try:
        with builder._hip_build_environment(tools, context.architectures) as cpp_extension:
            assert cpp_extension.ROCM_HOME == context.rocm_home
            assert cpp_extension.HIP_HOME == context.rocm_home + "/hip"
            assert cpp_extension.IS_HIP_EXTENSION is True
            assert os.environ["ROCM_HOME"] == context.rocm_home
            # Use PyTorch's real linkage preparation, not a mocked load result.
            options = dict(extra_ldflags=[], with_cuda=True, with_sycl=False, verbose=False, is_standalone=False)
            signature = inspect.signature(cpp_extension._prepare_ldflags)
            flags = cpp_extension._prepare_ldflags(**{{k:v for k,v in options.items() if k in signature.parameters}})
            assert "-lc10_hip" in flags and "-ltorch_hip" in flags
            assert "-lamdhip64" in flags
            assert "-lc10_cuda" not in flags and "-ltorch_cuda" not in flags
            assert "-lcudart" not in flags
            if {fail_inside!r}:
                raise RuntimeError("simulated build failure")
    except RuntimeError as exc:
        assert {fail_inside!r} and str(exc) == "simulated build failure"
    assert os.environ == before_env
    assert cpp_extension.ROCM_HOME is None
    assert cpp_extension.HIP_HOME is None
    assert cpp_extension.IS_HIP_EXTENSION is False
    assert "exllamav3" not in sys.modules
print("HIP linkage selected; environment and module state restored")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HIP linkage selected" in result.stdout


def test_explicit_prebuilt_extension_short_circuits_build(monkeypatch, tmp_path):
    extension = tmp_path / "prebuilt.so"
    extension.touch()
    expected = SimpleNamespace(reconstruct=lambda: None)
    monkeypatch.setattr(rocm_build.torch.version, "hip", "7.1")
    monkeypatch.setenv("UNSLOTH_EXL3_ROCM_EXTENSION", str(extension))
    monkeypatch.setattr(rocm_build, "load_python_extension", lambda path, label: expected)
    monkeypatch.setattr(
        rocm_build,
        "make_build_context",
        lambda: pytest.fail("automatic build should not run for an explicit prebuilt"),
    )

    assert rocm_build.load_or_build_extension() is expected


def test_missing_reconstruct_entry_point_is_rejected():
    with pytest.raises(rocm_build.RocmExtensionError, match="no callable reconstruct"):
        rocm_build._validate_module(SimpleNamespace(), label="test extension")


def test_compatible_cache_is_loaded_without_compiling(monkeypatch, tmp_path):
    context = _context()
    build_dir = tmp_path / context.key
    build_dir.mkdir()
    library = build_dir / f"{context.module_name}.so"
    library.touch()
    (build_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key": context.key,
                "context": rocm_build._context_payload(context),
                "library": library.name,
            }
        ),
        encoding="utf-8",
    )
    expected = SimpleNamespace(reconstruct=lambda: None)
    monkeypatch.setattr(rocm_build.torch.version, "hip", "7.1")
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_EXTENSION", raising=False)
    monkeypatch.delenv("EXL3_ROCM_BUILD_DIR", raising=False)
    monkeypatch.setattr(rocm_build, "make_build_context", lambda: (context, tmp_path))
    monkeypatch.setattr(rocm_build, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(rocm_build, "load_python_extension", lambda path, label: expected)
    monkeypatch.setattr(
        rocm_build,
        "resolve_build_tools",
        lambda *args: pytest.fail("cache hit must not resolve compiler or Ninja"),
    )
    monkeypatch.setattr(
        rocm_build.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("cache hit must not execute hipcc"),
    )
    monkeypatch.setattr(
        rocm_build,
        "resolve_ninja",
        lambda: pytest.fail("cache hit must not require Ninja"),
    )
    monkeypatch.setattr(
        rocm_build,
        "_compile_extension",
        lambda *args: pytest.fail("compatible cache should not compile"),
    )

    assert rocm_build.load_or_build_extension() is expected


def test_incompatible_cache_metadata_is_rejected(monkeypatch, tmp_path):
    context = _context()
    build_dir = tmp_path / context.key
    build_dir.mkdir()
    (build_dir / "manifest.json").write_text(
        json.dumps({"key": "wrong", "context": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rocm_build.torch.version, "hip", "7.1")
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_EXTENSION", raising=False)
    monkeypatch.delenv("EXL3_ROCM_BUILD_DIR", raising=False)
    monkeypatch.setattr(rocm_build, "make_build_context", lambda: (context, tmp_path))
    monkeypatch.setattr(rocm_build, "cache_root", lambda: tmp_path)

    with pytest.raises(rocm_build.RocmExtensionError, match="incompatible cached"):
        rocm_build.load_or_build_extension()


def test_successful_build_writes_compatible_manifest(monkeypatch, tmp_path):
    context = _context()
    module = SimpleNamespace(reconstruct=lambda: None)
    tools = rocm_build.BuildTools(
        rocm_home=Path(context.rocm_home),
        hipcc=Path(context.hipcc),
        hipcc_version="HIP version: 7.1.1",
        ninja=tmp_path / "ninja",
    )

    def fake_compile(build_context, source_dir, build_tools, build_dir):
        assert build_context == context
        assert build_tools == tools
        (build_dir / f"{context.module_name}.so").touch()
        return module

    monkeypatch.setattr(rocm_build.torch.version, "hip", "7.1")
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_EXTENSION", raising=False)
    monkeypatch.delenv("EXL3_ROCM_BUILD_DIR", raising=False)
    monkeypatch.setattr(rocm_build, "make_build_context", lambda: (context, tmp_path))
    monkeypatch.setattr(rocm_build, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(rocm_build, "resolve_build_tools", lambda build_context: tools)
    monkeypatch.setattr(rocm_build, "_compile_extension", fake_compile)

    assert rocm_build.load_or_build_extension() is module
    manifest = json.loads((tmp_path / context.key / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["key"] == context.key
    assert manifest["context"] == rocm_build._context_payload(context)
    library_path = Path(manifest["library"])
    assert library_path.parts[:1] == ("attempts",)
    assert library_path.name == f"{context.module_name}.so"
    assert manifest["build_provenance"] == {
        "hipcc": str(tools.hipcc),
        "hipcc_version": tools.hipcc_version,
        "ninja": str(tools.ninja),
    }


def test_compilation_failure_reports_build_inputs(monkeypatch, tmp_path):
    context = _context()
    tools = rocm_build.BuildTools(
        rocm_home=Path(context.rocm_home),
        hipcc=Path(context.hipcc),
        hipcc_version="HIP version: 7.1.1",
        ninja=tmp_path / "ninja",
    )
    monkeypatch.setattr(rocm_build.torch.version, "hip", "7.1")
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_EXTENSION", raising=False)
    monkeypatch.delenv("EXL3_ROCM_BUILD_DIR", raising=False)
    monkeypatch.setattr(rocm_build, "make_build_context", lambda: (context, tmp_path))
    monkeypatch.setattr(rocm_build, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(rocm_build, "resolve_build_tools", lambda build_context: tools)
    monkeypatch.setattr(
        rocm_build,
        "_compile_extension",
        lambda *args: (_ for _ in ()).throw(ValueError("compiler stopped")),
    )

    with pytest.raises(rocm_build.RocmExtensionError, match="compilation failed") as error:
        rocm_build.load_or_build_extension()

    message = str(error.value)
    assert context.hipcc in message
    assert "gfx1100" in message
    assert "compiler stopped" in message


def test_abandoned_lock_file_is_reusable(tmp_path):
    lock_path = tmp_path / ".locks" / "stale.lock"
    lock_path.parent.mkdir()
    lock_path.write_text("previous process metadata\n", encoding="utf-8")

    with rocm_build._cache_build_lock(lock_path):
        assert lock_path.is_file()


def test_abandoned_torch_build_lock_uses_new_attempt(monkeypatch, tmp_path):
    context = _context()
    build_dir = tmp_path / context.key
    abandoned = build_dir / "attempts" / "attempt-abandoned"
    abandoned.mkdir(parents=True)
    (abandoned / "lock").write_text("", encoding="utf-8")
    module = SimpleNamespace(reconstruct=lambda: None)
    tools = rocm_build.BuildTools(
        rocm_home=Path(context.rocm_home),
        hipcc=Path(context.hipcc),
        hipcc_version="HIP version: 7.1.1",
        ninja=tmp_path / "ninja",
    )
    observed_attempts = []

    def fake_compile(build_context, source_dir, build_tools, attempt_dir):
        observed_attempts.append(attempt_dir)
        assert attempt_dir != abandoned
        assert not (attempt_dir / "lock").exists()
        (attempt_dir / f"{context.module_name}.so").touch()
        return module

    monkeypatch.setattr(rocm_build.torch.version, "hip", "7.1")
    monkeypatch.delenv("UNSLOTH_EXL3_ROCM_EXTENSION", raising=False)
    monkeypatch.delenv("EXL3_ROCM_BUILD_DIR", raising=False)
    monkeypatch.setattr(rocm_build, "make_build_context", lambda: (context, tmp_path))
    monkeypatch.setattr(rocm_build, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(rocm_build, "resolve_build_tools", lambda build_context: tools)
    monkeypatch.setattr(rocm_build, "_compile_extension", fake_compile)

    assert rocm_build.load_or_build_extension() is module
    assert len(observed_attempts) == 1
    assert (abandoned / "lock").is_file()


def test_cli_explicit_prebuilt_needs_no_automatic_build_inputs(monkeypatch, tmp_path, capsys):
    extension = tmp_path / "prebuilt.so"
    extension.touch()
    module = SimpleNamespace(reconstruct=lambda: None)
    info = rocm_build.ExtensionLoadInfo(
        module=module,
        origin="explicit prebuilt",
        extension_path=extension,
    )
    monkeypatch.setattr(rocm_build, "load_or_build_extension_info", lambda: info)
    monkeypatch.setattr(
        rocm_build,
        "make_build_context",
        lambda: pytest.fail("CLI must not identify an automatic build after prebuilt load"),
    )
    monkeypatch.setattr(
        rocm_build,
        "resolve_build_tools",
        lambda *args: pytest.fail("CLI must not require compiler or Ninja"),
    )

    rocm_build.main()

    output = capsys.readouterr().out
    assert "PASS: EXL3 ROCm reconstruction extension is ready" in output
    assert "Origin: explicit prebuilt" in output
    assert f"Extension: {extension}" in output


def test_two_processes_publish_one_fresh_build(tmp_path):
    """Two fresh callers serialize staging/compile/publication by cache key."""

    cache_dir = tmp_path / "cache"
    source_dir = _source_tree(tmp_path / "source")
    script = tmp_path / "concurrent_caller.py"
    compile_log = tmp_path / "compile.log"
    overlap = tmp_path / "overlap"
    active = tmp_path / "compile.active"
    start = tmp_path / "start"
    script.write_text(
        f"""
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

module_path = Path({str(MODULE_PATH)!r})
spec = importlib.util.spec_from_file_location("_concurrent_rocm_build", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.torch.version.hip = "7.1"

context = module.BuildContext(
    schema=2,
    torch_version="2.10.0+rocm7.1",
    hip_version="7.1",
    python_cache_tag="cpython-313",
    python_soabi="cpython-313-x86_64-linux-gnu",
    cxx11_abi="True",
    machine="x86_64",
    architectures=("gfx1100",),
    rocm_home="/opt/rocm-test",
    hipcc="/opt/rocm-test/bin/hipcc",
    source_fingerprint="a" * 64,
)
tools = module.BuildTools(
    rocm_home=Path(context.rocm_home),
    hipcc=Path(context.hipcc),
    hipcc_version="HIP version: test",
    ninja=Path("/ninja"),
)
module._explicit_prebuilt = lambda: None
module.make_build_context = lambda: (context, Path({str(source_dir)!r}))
module.cache_root = lambda: Path({str(cache_dir)!r})
module.resolve_build_tools = lambda build_context: tools
module.load_python_extension = lambda path, label: SimpleNamespace(reconstruct=lambda: None)

def compile_once(build_context, native_source, build_tools, build_dir):
    try:
        fd = os.open({str(active)!r}, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        Path({str(overlap)!r}).write_text("overlap\\n", encoding="utf-8")
        raise
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        with Path({str(compile_log)!r}).open("a", encoding="utf-8") as handle:
            handle.write(str(os.getpid()) + "\\n")
        time.sleep(0.5)
        (build_dir / f"{{context.module_name}}.so").touch()
        return SimpleNamespace(reconstruct=lambda: None)
    finally:
        Path({str(active)!r}).unlink(missing_ok=True)

module._compile_extension = compile_once
while not Path({str(start)!r}).exists():
    time.sleep(0.01)
info = module.load_or_build_extension_info()
print(info.origin)
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    processes = [
        subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(2)
    ]
    time.sleep(0.2)
    start.touch()
    results = [process.communicate(timeout=20) for process in processes]

    for process, (stdout, stderr) in zip(processes, results):
        assert process.returncode == 0, stderr
        assert stdout.strip() in {"built", "cache"}
    assert sorted(stdout.strip() for stdout, _ in results) == ["built", "cache"]
    assert compile_log.read_text(encoding="utf-8").count("\n") == 1
    assert not overlap.exists()
