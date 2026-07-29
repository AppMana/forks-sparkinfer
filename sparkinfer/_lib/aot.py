"""Ahead-of-time CuTe-DSL kernel objects shipped inside the wheel.

Invariant this module exists to enforce: **a CuTe-DSL kernel that production
launches must already be compiled when the wheel is installed.**  A
``cute.compile`` that runs inside a live request is a multi-second stall in the
serving path, needs a writable cache on the serving node, and is exactly the
failure the PCIe extension AOT build (setup.py) already removed for the C++
side.

How AOT works for the DSL, and why it works
-------------------------------------------
CUTLASS 4.6 supports true ahead-of-time export.  Verified against the
installed package:

* ``JitCompiledFunction.dump_to_object(prefix)``
  (``cutlass/base_dsl/jit_executor.py:1456``) serializes a compiled function
  to an ELF object containing the host launch entry *and* the cubin.
* ``ExternalBinaryModule(path)``
  (``cutlass/base_dsl/export/external_binary_module.py:70``) loads such an
  object in **any** process and ``__getattr__`` returns a runnable
  ``JitCompiledFunction``.  It re-derives the signature from metadata encoded
  into the object, so nothing is keyed on pointer values or on JIT-time
  process state.
* ``ExternalBinaryModule.__getattr__`` calls ``load_provider.version_checker``
  on the object-file version embedded at export time, so a load against a
  different CUTLASS is rejected rather than silently mis-run.
* The GPU architecture that is code-generated comes from ``CUTE_DSL_ARCH``
  when set, and only otherwise from the live device
  (``cutlass/base_dsl/env_manager.py:222-246``).  ``Arch.sm_121a`` exists
  under CUDA 13 (``cutlass/base_dsl/arch.py:53``).

Measured, not assumed: compiling with ``CUDA_VISIBLE_DEVICES=""`` and
``CUTE_DSL_ARCH=sm_86`` produced a 10120-byte object, and a *separate* process
loaded it with ``ExternalBinaryModule`` and ran it correctly on a real
sm_86 device.  Cross-process, cross-machine reuse of compiled DSL kernels is
therefore sound.

What is shipped
---------------
``sparkinfer/_aot_cache/`` uses exactly the layout of the writable compile
cache in ``sparkinfer._lib.compiler``: ``<key[:2]>/<key>.o`` plus a ``.json``
manifest.  The cache key already covers the compile spec, the CUTLASS/torch
toolchain and the GPU arch, so objects for several architectures coexist in one
directory and only the matching ones can ever be selected.

Two properties of the key had to be repaired for a *shipped* cache to be
usable at all; both live in ``compiler.py`` and are load-bearing here:

1. The key includes a content fingerprint of every file under ``sparkinfer/``.
   Adding artifacts under that root would change the fingerprint and invalidate
   the artifacts being added.  ``_iter_fingerprint_files`` therefore excludes
   this directory.
2. The key included the *raw* ``CUTE_DSL_ARCH`` environment variable.  The
   builder sets it (it has no GPU); the serving node does not (it has one).
   Same kernel, same SASS, different key -- a guaranteed miss.  The key now
   carries the *resolved* architecture instead.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from threading import RLock

# Directory name under the installed ``sparkinfer`` package.  Referenced by
# setup.py (packaging), scripts/build_aot_cache.py (staging) and
# .github/workflows/wheels.yaml (the in-wheel assertion); keep the three in
# lockstep.
AOT_CACHE_DIRNAME = "_aot_cache"

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

_STATS_LOCK = RLock()
_AOT_HITS = 0
_JIT_MISSES = 0
_WARNED_KERNELS: set[str] = set()


def packaged_cache_dir() -> Path:
    """Read-only AOT cache directory shipped inside the installed package."""
    return _PACKAGE_ROOT / AOT_CACHE_DIRNAME


def _extra_cache_dirs() -> tuple[Path, ...]:
    raw = os.environ.get("SPARKINFER_AOT_CACHE_DIR", "")
    return tuple(Path(part) for part in raw.split(os.pathsep) if part)


def readonly_cache_dirs() -> tuple[Path, ...]:
    """Directories searched for prebuilt objects, highest priority first.

    ``SPARKINFER_AOT_CACHE_DIR`` (``os.pathsep``-separated) comes first so an
    operator can stage a newer cache without reinstalling the wheel; the
    in-wheel directory is the fallback.  Neither is ever written to.
    """
    dirs = [*_extra_cache_dirs(), packaged_cache_dir()]
    return tuple(d for d in dirs if d.is_dir())


def have_packaged_cache() -> bool:
    return packaged_cache_dir().is_dir()


def require_aot() -> bool:
    """True when a JIT compile must raise instead of proceeding.

    Defaults to off so a developer checkout stays usable.  Serving images set
    ``SPARKINFER_REQUIRE_AOT=1``: a JIT compile there is a bug, not a slow
    path, and it must surface at the launch that caused it rather than as an
    unexplained multi-second first-token latency.
    """
    return os.environ.get("SPARKINFER_REQUIRE_AOT", "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _warn_once_key(kernel_id: str, cache_key: str) -> str:
    return kernel_id or cache_key[:16]


def record_aot_hit() -> None:
    global _AOT_HITS
    with _STATS_LOCK:
        _AOT_HITS += 1


def note_jit_compile(*, kernel_id: str, cache_key: str, detail: str) -> None:
    """Report that a launch is about to JIT because the AOT cache missed.

    Loud by construction: a warning on the first miss per kernel id, and a hard
    error under ``SPARKINFER_REQUIRE_AOT``.  Silence here is what lets a
    multi-second ``cute.compile`` hide inside the first live request.
    """
    global _JIT_MISSES
    with _STATS_LOCK:
        _JIT_MISSES += 1
        key = _warn_once_key(kernel_id, cache_key)
        first = key not in _WARNED_KERNELS
        _WARNED_KERNELS.add(key)

    message = (
        f"sparkinfer is JIT-compiling a CuTe-DSL kernel that the wheel does not "
        f"carry ahead of time: kernel={kernel_id or '<unkeyed>'} "
        f"cache_key={cache_key[:16]} {detail}. "
        f"packaged_aot_cache={'present' if have_packaged_cache() else 'ABSENT'}. "
        "Add this configuration to sparkinfer/_lib/aot_matrix.py and rebuild the "
        "wheel, or stage a cache via SPARKINFER_AOT_CACHE_DIR."
    )
    if require_aot():
        raise AotCacheMiss(message)
    if first:
        warnings.warn(message, AotCacheMissWarning, stacklevel=3)


class AotCacheMiss(RuntimeError):
    """A launch needed a kernel the AOT cache does not carry."""


class AotCacheMissWarning(UserWarning):
    """Emitted once per kernel when a launch falls back to JIT."""


def aot_info() -> dict[str, object]:
    with _STATS_LOCK:
        return {
            "packaged_cache_dir": str(packaged_cache_dir()),
            "packaged_cache_present": have_packaged_cache(),
            "readonly_cache_dirs": [str(d) for d in readonly_cache_dirs()],
            "require_aot": require_aot(),
            "aot_hits": _AOT_HITS,
            "jit_compiles": _JIT_MISSES,
            "jit_kernels": sorted(_WARNED_KERNELS),
        }


def reset_aot_stats() -> None:
    global _AOT_HITS, _JIT_MISSES
    with _STATS_LOCK:
        _AOT_HITS = 0
        _JIT_MISSES = 0
        _WARNED_KERNELS.clear()


@lru_cache(maxsize=1)
def resolved_gpu_arch() -> str:
    """The architecture CuTe-DSL will actually code-generate for.

    ``CUTE_DSL_ARCH`` wins when set, exactly as
    ``cutlass/base_dsl/env_manager.py:222-246`` resolves it; otherwise the live
    device's compute capability is used, with the same ``a`` suffix rule
    (``major >= 9``).  Returning the *resolved* value rather than the raw
    environment variable is what makes a cache built on a GPU-less runner
    (``CUTE_DSL_ARCH=sm_121a``) hit on a Spark that sets nothing.

    ``unknown`` when neither source answers: it is distinct from every real
    arch, so it can only ever miss, never mis-hit.
    """
    override = os.environ.get("CUTE_DSL_ARCH", "").strip()
    if override:
        return override
    try:
        from cutlass.base_dsl.runtime.cuda import get_compute_capability_major_minor

        major, minor = get_compute_capability_major_minor()
    except Exception:
        return "unknown"
    if major is None or minor is None:
        return "unknown"
    suffix = "a" if major >= 9 else ""
    return f"sm_{major}{minor}{suffix}"
