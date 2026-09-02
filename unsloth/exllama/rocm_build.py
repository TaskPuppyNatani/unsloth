# Copyright 2023-present Daniel Han-Chen & the Unsloth team.
#
# Licensed under the Apache License, Version 2.0.
"""Build and load the narrow EXL3 reconstruction extension on ROCm.

This module intentionally builds only the correctness-first reconstruction
kernel used by the ROCm LoRA path.  It does not import ExLlamaV3 or build its
full CUDA extension.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch


_BUILD_SCHEMA = 2
_EXTENSION_BASENAME = "exl3_rocm_reconstruct"
_DEFAULT_ROCM_HOME = Path("/opt/rocm")
_SOURCE_FILES = (
    "reconstruct_rocm.cpp",
    "reconstruct_rocm.cu",
    "reconstruct_rocm.h",
    "portable_decode_device.cuh",
    "portable_codebook.h",
    "portable_bitops.h",
)
_ARCH_RE = re.compile(r"^gfx[0-9a-f]+(?::[a-z0-9_+\-]+)*$", re.IGNORECASE)
# cpp_extension and os.environ are process-global even for different cache keys.
_TORCH_BUILD_LOCK = threading.RLock()


class RocmExtensionError(RuntimeError):
    """Actionable failure while locating, building, or loading the extension."""


@dataclass(frozen=True)
class BuildContext:
    schema: int
    torch_version: str
    hip_version: str
    python_cache_tag: str
    python_soabi: str
    cxx11_abi: str
    machine: str
    architectures: tuple[str, ...]
    rocm_home: str
    hipcc: str
    source_fingerprint: str

    @property
    def key(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def module_name(self) -> str:
        return f"{_EXTENSION_BASENAME}_{self.key[:20]}"


@dataclass(frozen=True)
class BuildTools:
    rocm_home: Path
    hipcc: Path
    hipcc_version: str
    ninja: Path


@dataclass(frozen=True)
class ExtensionLoadInfo:
    module: Any
    origin: str
    extension_path: Path
    context: BuildContext | None = None
    source_dir: Path | None = None
    build_dir: Path | None = None
    build_tools: BuildTools | None = None


def _context_payload(context: BuildContext) -> dict[str, Any]:
    """Return the JSON-stable representation persisted in cache manifests."""

    return json.loads(json.dumps(asdict(context), sort_keys=True))


def _require_rocm() -> str:
    hip_version = getattr(torch.version, "hip", None)
    if hip_version is None:
        raise RocmExtensionError(
            "Unsloth: EXL3 ROCm reconstruction requires a ROCm PyTorch "
            "build, but torch.version.hip is None. A CUDA or CPU PyTorch "
            "installation cannot build this HIP-only extension."
        )
    return str(hip_version)


def _split_architectures(value: str) -> tuple[str, ...]:
    values = re.split(r"[;,\s]+", value.strip())
    architectures = tuple(dict.fromkeys(item for item in values if item))
    invalid = [item for item in architectures if not _ARCH_RE.fullmatch(item)]
    if invalid:
        raise RocmExtensionError(
            "Unsloth: invalid PYTORCH_ROCM_ARCH value(s): "
            f"{', '.join(invalid)}. Expected values such as gfx1100, "
            "separated by semicolons."
        )
    return architectures


def detect_architectures() -> tuple[str, ...]:
    """Return explicit architectures or the active AMD device architecture."""

    explicit = os.environ.get("PYTORCH_ROCM_ARCH")
    if explicit:
        architectures = _split_architectures(explicit)
        if architectures:
            return architectures

    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False

    if available:
        try:
            device_index = int(torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(device_index)
            architecture = str(getattr(properties, "gcnArchName", "")).strip()
            if architecture:
                return _split_architectures(architecture)
        except RocmExtensionError:
            raise
        except Exception as exc:
            raise RocmExtensionError(
                "Unsloth: ROCm PyTorch found an AMD GPU but could not detect "
                f"its gcnArchName: {exc}. Set PYTORCH_ROCM_ARCH explicitly "
                "(for example, gfx1100)."
            ) from exc

    raise RocmExtensionError(
        "Unsloth: could not detect an active AMD GPU architecture. Ensure the "
        "ROCm GPU is visible to PyTorch or set PYTORCH_ROCM_ARCH explicitly "
        "(for example, gfx1100)."
    )


def _validate_source_dir(path: Path, *, packaged: bool = False) -> Path:
    path = path.expanduser().resolve()
    missing = [name for name in _SOURCE_FILES if not (path / name).is_file()]
    if missing:
        if packaged:
            raise RocmExtensionError(
                "Unsloth: packaged EXL3 ROCm reconstruction sources are "
                f"incomplete at {path}; missing: {', '.join(missing)}. "
                "Reinstall Unsloth from a distribution that includes "
                "unsloth/exllama/rocm_ext, or set "
                "UNSLOTH_EXL3_ROCM_SOURCE_DIR to a complete advanced/debug "
                "source override."
            )
        raise RocmExtensionError(
            f"Unsloth: EXL3 ROCm source directory {path} is incomplete; "
            f"missing: {', '.join(missing)}."
        )
    return path


def _packaged_source_dir() -> Path:
    """Return the native-source directory shipped beside this module."""

    return Path(__file__).resolve().parent / "rocm_ext"


def find_source_dir() -> Path:
    """Locate an explicit override or Unsloth's packaged native sources."""

    override = os.environ.get("UNSLOTH_EXL3_ROCM_SOURCE_DIR")
    if override:
        return _validate_source_dir(Path(override))

    return _validate_source_dir(_packaged_source_dir(), packaged=True)


def source_fingerprint(source_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema={_BUILD_SCHEMA}\n".encode("ascii"))
    digest.update(b"rocm_build.py\0")
    digest.update(Path(__file__).read_bytes())
    digest.update(b"\0")
    for name in _SOURCE_FILES:
        path = source_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def select_rocm_home() -> Path:
    """Select a toolchain root without requiring the compiler to be present."""

    override = os.environ.get("UNSLOTH_EXL3_ROCM_HOME")
    if override:
        return Path(override).expanduser().resolve()

    for variable in ("ROCM_HOME", "ROCM_PATH"):
        value = os.environ.get(variable)
        if value:
            return Path(value).expanduser().resolve()

    # Keep the default stable so a compatible cache remains addressable even
    # when build-only tools are temporarily unavailable. Non-standard ROCm
    # layouts should set UNSLOTH_EXL3_ROCM_HOME explicitly.
    return _DEFAULT_ROCM_HOME.resolve()


def resolve_ninja() -> Path:
    override = os.environ.get("UNSLOTH_EXL3_ROCM_NINJA")
    if override:
        candidate = Path(override).expanduser().resolve()
    else:
        found = shutil.which("ninja")
        candidate = Path(found).resolve() if found else Path("ninja")

    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RocmExtensionError(
            "Unsloth: Ninja is required to build the EXL3 ROCm extension but "
            "was not found. Set UNSLOTH_EXL3_ROCM_NINJA to the ninja "
            "executable or add ninja to PATH."
        )
    return candidate


def make_build_context() -> tuple[BuildContext, Path]:
    """Identify a cache without executing or requiring build-only tools."""

    hip_version = _require_rocm()
    source_dir = find_source_dir()
    rocm_home = select_rocm_home()
    hipcc = rocm_home / "bin" / "hipcc"
    architectures = detect_architectures()
    context = BuildContext(
        schema=_BUILD_SCHEMA,
        torch_version=str(torch.__version__),
        hip_version=hip_version,
        python_cache_tag=str(getattr(sys.implementation, "cache_tag", "unknown")),
        python_soabi=str(sysconfig.get_config_var("SOABI") or "unknown"),
        cxx11_abi=str(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown")),
        machine=platform.machine(),
        architectures=architectures,
        rocm_home=str(rocm_home),
        hipcc=str(hipcc),
        source_fingerprint=source_fingerprint(source_dir),
    )
    return context, source_dir


def resolve_build_tools(context: BuildContext) -> BuildTools:
    """Validate and identify tools only after a cache miss owns the build lock."""

    rocm_home = Path(context.rocm_home)
    hipcc = Path(context.hipcc)
    if not rocm_home.is_dir():
        raise RocmExtensionError(
            "Unsloth: a usable HIP compiler/toolchain was not found. The "
            f"configured ROCm root does not exist: {rocm_home}. Set "
            "UNSLOTH_EXL3_ROCM_HOME to a ROCm root containing bin/hipcc."
        )
    if not hipcc.is_file() or not os.access(hipcc, os.X_OK):
        raise RocmExtensionError(
            "Unsloth: a usable HIP compiler/toolchain was not found. "
            f"{hipcc} is missing or not executable. Set "
            "UNSLOTH_EXL3_ROCM_HOME to a ROCm root containing bin/hipcc."
        )
    try:
        result = subprocess.run(
            [str(hipcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        raise RocmExtensionError(
            f"Unsloth: HIP compiler probe failed: {hipcc} --version: {exc}"
        ) from exc
    version_text = "\n".join((result.stdout, result.stderr)).strip()
    version_line = next(
        (line.strip() for line in version_text.splitlines() if line.strip()),
        "unknown hipcc version",
    )
    return BuildTools(
        rocm_home=rocm_home,
        hipcc=hipcc,
        hipcc_version=version_line,
        ninja=resolve_ninja(),
    )


def cache_root() -> Path:
    override = os.environ.get("UNSLOTH_EXL3_ROCM_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "unsloth" / "exl3-rocm").resolve()


def _module_name(path: Path) -> str:
    name = path.name
    if ".cpython-" in name:
        return name.split(".cpython-", 1)[0]
    if ".abi3" in name:
        return name.split(".abi3", 1)[0]
    return name.rsplit(".so", 1)[0]


def load_python_extension(path: Path, *, label: str) -> Any:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RocmExtensionError(f"Unsloth: {label} does not exist: {path}")
    try:
        spec = importlib.util.spec_from_file_location(_module_name(path), path)
        if spec is None or spec.loader is None:
            raise ImportError("could not create an import specification")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RocmExtensionError(
            f"Unsloth: {label} is incompatible or could not be loaded: "
            f"{path}\n{type(exc).__name__}: {exc}"
        ) from exc
    if not callable(getattr(module, "reconstruct", None)):
        raise RocmExtensionError(
            f"Unsloth: {label} has no callable reconstruct() entry point: {path}"
        )
    return module


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RocmExtensionError(
            f"Unsloth: incompatible cached EXL3 ROCm extension metadata at "
            f"{path}: {exc}. Select a fresh cache with "
            "UNSLOTH_EXL3_ROCM_CACHE_DIR."
        ) from exc
    if not isinstance(value, dict):
        raise RocmExtensionError(
            f"Unsloth: incompatible cached EXL3 ROCm extension metadata at "
            f"{path}: expected a JSON object."
        )
    return value


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def _cache_build_lock(path: Path) -> Iterator[None]:
    """Hold an advisory process lock for one compatibility-keyed publication."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - ROCm is supported on Linux.
        raise RocmExtensionError(
            "Unsloth: EXL3 ROCm cache publication requires POSIX file locks."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("a+b")
    except Exception as exc:
        raise RocmExtensionError(
            f"Unsloth: could not open EXL3 ROCm cache lock {path}: {exc}"
        ) from exc

    with handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                break
            except InterruptedError:
                continue
        try:
            yield
        finally:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    break
                except InterruptedError:
                    continue


@contextmanager
def _build_environment(
    rocm_home: Path,
    ninja: Path,
    architectures: tuple[str, ...],
) -> Iterator[None]:
    updates = {
        "ROCM_HOME": str(rocm_home),
        "ROCM_PATH": str(rocm_home),
        "PATH": os.pathsep.join(
            (str(ninja.parent), str(rocm_home / "bin"), os.environ.get("PATH", ""))
        ),
        "PYTORCH_ROCM_ARCH": ";".join(architectures),
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _hip_build_environment(tools: BuildTools, architectures: tuple[str, ...]) -> Iterator[Any]:
    """Temporarily select HIP in fresh or previously imported PyTorch utilities."""

    _require_rocm()
    with _TORCH_BUILD_LOCK:
        # Import before changing the environment so even a fresh import's
        # original discovery state can be restored after this build.
        from torch.utils import cpp_extension

        updates = {
            "ROCM_HOME": str(tools.rocm_home),
            "HIP_HOME": str(tools.rocm_home / "hip"),
            "IS_HIP_EXTENSION": True,
        }
        previous = {name: getattr(cpp_extension, name) for name in updates}
        try:
            with _build_environment(tools.rocm_home, tools.ninja, architectures):
                for name, value in updates.items():
                    setattr(cpp_extension, name, value)
                yield cpp_extension
        finally:
            for name, value in previous.items():
                setattr(cpp_extension, name, value)


def _compile_extension(
    context: BuildContext,
    source_dir: Path,
    tools: BuildTools,
    build_dir: Path,
) -> Any:
    # PyTorch hipify can place generated siblings beside source files supplied
    # from outside build_directory. Stage an exact copy in the compatibility
    # cache so a first build never writes generated *_hip files to the checkout.
    staged_source_dir = build_dir / "source"
    staged_source_dir.mkdir(parents=True, exist_ok=True)
    for name in _SOURCE_FILES:
        shutil.copyfile(source_dir / name, staged_source_dir / name)

    with _hip_build_environment(tools, context.architectures) as cpp_extension:
        return cpp_extension.load(
            name=context.module_name,
            sources=[
                str(staged_source_dir / "reconstruct_rocm.cpp"),
                str(staged_source_dir / "reconstruct_rocm.cu"),
            ],
            extra_include_paths=[str(staged_source_dir)],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2"],
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=os.environ.get("UNSLOTH_EXL3_ROCM_VERBOSE", "0") == "1",
        )


def _validate_module(module: Any, *, label: str) -> Any:
    if not callable(getattr(module, "reconstruct", None)):
        raise RocmExtensionError(
            f"Unsloth: {label} loaded but has no callable reconstruct() entry point."
        )
    return module


def _explicit_prebuilt() -> ExtensionLoadInfo | None:
    exact = os.environ.get("UNSLOTH_EXL3_ROCM_EXTENSION")
    if exact:
        path = Path(exact).expanduser().resolve()
        return ExtensionLoadInfo(
            module=load_python_extension(
                path,
                label="explicit EXL3 ROCm extension",
            ),
            origin="explicit prebuilt",
            extension_path=path,
        )

    legacy_dir = os.environ.get("EXL3_ROCM_BUILD_DIR")
    if not legacy_dir:
        return None
    candidates = sorted(Path(legacy_dir).expanduser().glob("*.so"))
    if not candidates:
        return None
    errors: list[str] = []
    for candidate in candidates:
        try:
            path = candidate.resolve()
            return ExtensionLoadInfo(
                module=load_python_extension(
                    path,
                    label="legacy EXL3 ROCm extension",
                ),
                origin="legacy prebuilt",
                extension_path=path,
            )
        except RocmExtensionError as exc:
            errors.append(str(exc))
    raise RocmExtensionError("\n".join(errors))


def _load_cached_extension(
    context: BuildContext,
    source_dir: Path,
    build_dir: Path,
) -> ExtensionLoadInfo | None:
    manifest_path = build_dir / "manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = _read_manifest(manifest_path)
    expected_context = _context_payload(context)
    if manifest.get("key") != context.key or manifest.get("context") != expected_context:
        raise RocmExtensionError(
            "Unsloth: incompatible cached EXL3 ROCm extension metadata at "
            f"{manifest_path}. The cache does not match this PyTorch/ROCm/"
            "Python/architecture/source combination. Select a fresh cache "
            "with UNSLOTH_EXL3_ROCM_CACHE_DIR."
        )
    library_name = manifest.get("library")
    if not isinstance(library_name, str):
        raise RocmExtensionError(
            f"Unsloth: incompatible cached EXL3 ROCm extension metadata at "
            f"{manifest_path}: invalid library name."
        )
    library_path = Path(library_name)
    if library_path.is_absolute() or ".." in library_path.parts:
        raise RocmExtensionError(
            f"Unsloth: incompatible cached EXL3 ROCm extension metadata at "
            f"{manifest_path}: invalid library path."
        )
    extension_path = (build_dir / library_path).resolve()
    try:
        extension_path.relative_to(build_dir.resolve())
    except ValueError as exc:
        raise RocmExtensionError(
            f"Unsloth: incompatible cached EXL3 ROCm extension metadata at "
            f"{manifest_path}: library resolves outside the build cache."
        ) from exc
    return ExtensionLoadInfo(
        module=load_python_extension(
            extension_path,
            label="cached EXL3 ROCm extension",
        ),
        origin="cache",
        extension_path=extension_path,
        context=context,
        source_dir=source_dir,
        build_dir=build_dir,
    )


def load_or_build_extension_info() -> ExtensionLoadInfo:
    """Load an override/cached extension, or publish one under a process lock."""

    _require_rocm()
    prebuilt = _explicit_prebuilt()
    if prebuilt is not None:
        return prebuilt

    context, source_dir = make_build_context()
    root = cache_root()
    build_dir = root / context.key
    manifest_path = build_dir / "manifest.json"
    cached = _load_cached_extension(context, source_dir, build_dir)
    if cached is not None:
        return cached

    lock_path = root / ".locks" / f"{context.key}.lock"
    with _cache_build_lock(lock_path):
        # A waiter must re-check after acquiring the publication lock. The
        # winning process atomically publishes the manifest only after staging,
        # compiling, selecting the library, and validating its entry point.
        cached = _load_cached_extension(context, source_dir, build_dir)
        if cached is not None:
            return cached

        try:
            build_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RocmExtensionError(
                f"Unsloth: could not create EXL3 ROCm build cache {build_dir}: {exc}"
            ) from exc

        tools = resolve_build_tools(context)
        attempts_dir = build_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        # A dead torch.utils.cpp_extension.load process can leave its own
        # FileBaton lock behind. Each outer-lock owner gets a fresh attempt
        # directory, so that stale inner lock cannot block recovery. Attempts
        # are intentionally retained: this path never deletes build artifacts.
        attempt_dir = Path(tempfile.mkdtemp(prefix=f"attempt-{os.getpid()}-", dir=attempts_dir))
        try:
            module = _validate_module(
                _compile_extension(context, source_dir, tools, attempt_dir),
                label="newly built EXL3 ROCm extension",
            )
        except RocmExtensionError:
            raise
        except Exception as exc:
            raise RocmExtensionError(
                "Unsloth: EXL3 ROCm extension compilation failed.\n"
                f"Source: {source_dir}\n"
                f"Build cache: {build_dir}\n"
                f"HIP compiler: {tools.hipcc}\n"
                f"Architectures: {';'.join(context.architectures)}\n"
                f"PyTorch: {context.torch_version}\n"
                f"ROCm runtime: {context.hip_version}\n"
                f"{type(exc).__name__}: {exc}"
            ) from exc

        libraries = sorted(attempt_dir.glob(f"{context.module_name}*.so"))
        if not libraries:
            raise RocmExtensionError(
                "Unsloth: EXL3 ROCm extension compiled and loaded, but its shared "
                f"library was not found in {attempt_dir}."
            )
        library = libraries[-1]
        _write_manifest(
            manifest_path,
            {
                "key": context.key,
                "context": _context_payload(context),
                "library": library.relative_to(build_dir).as_posix(),
                "source_dir": str(source_dir),
                "build_provenance": {
                    "hipcc": str(tools.hipcc),
                    "hipcc_version": tools.hipcc_version,
                    "ninja": str(tools.ninja),
                },
            },
        )
        return ExtensionLoadInfo(
            module=module,
            origin="built",
            extension_path=library,
            context=context,
            source_dir=source_dir,
            build_dir=build_dir,
            build_tools=tools,
        )


def load_or_build_extension() -> Any:
    """Return the extension module used by the reconstruction provider."""

    return load_or_build_extension_info().module


def main() -> None:
    info = load_or_build_extension_info()
    print("PASS: EXL3 ROCm reconstruction extension is ready")
    print(f"Origin: {info.origin}")
    print(f"Extension: {info.extension_path}")
    if info.source_dir is not None:
        print(f"Source: {info.source_dir}")
    if info.build_dir is not None:
        print(f"Cache: {info.build_dir}")
    if info.context is not None:
        print(f"Architectures: {';'.join(info.context.architectures)}")
    entry_point = getattr(info.module.reconstruct, "__name__", "reconstruct")
    print(f"Entry point: {entry_point}")


if __name__ == "__main__":
    main()
