#!/usr/bin/env python3
"""Stage prebuilt CuTe-DSL kernel objects into the package for the wheel.

Why this is a capture-and-stage step and not a self-contained compiler
---------------------------------------------------------------------
CUTLASS 4.6 fully supports ahead-of-time export, and it does not need a GPU to
do it.  Verified against the installed package:
``JitCompiledFunction.dump_to_object`` (``cutlass/base_dsl/jit_executor.py:1456``)
writes an ELF carrying the host launcher and the cubin;
``ExternalBinaryModule`` (``cutlass/base_dsl/export/external_binary_module.py:70``)
loads it in a different process; ``CUTE_DSL_ARCH`` selects the target
architecture without consulting a device
(``cutlass/base_dsl/env_manager.py:222-246``).  Measured: compiling under
``CUDA_VISIBLE_DEVICES="" CUTE_DSL_ARCH=sm_86`` produced an object that a
separate process loaded and ran correctly on a real sm_86 GPU.

What is *not* device-free is reaching sparkinfer's ``compile()`` calls.  Every
op derives its ``KernelCompileSpec`` from live tensors inside ``run()`` -- the
compile arguments themselves are already fake pointers
(``attention/_shared/contiguous/api.py:1052``), but the spec that selects them
is computed from real shapes, dtypes, devices and from plan/bind scratch that
is really allocated.  Driving that without a device would mean refactoring
every op to expose a config-to-spec function.  Rather than claim a GPU-less
build that does not exist, this script captures the cache from a real warmup on
target hardware and ships what the warmup produced.

Usage
-----
  # 1. On a machine with the target GPU: run any warmup that exercises the
  #    production shapes -- normally the vLLM engine's own startup and CUDA
  #    graph capture -- with the compile cache redirected to a staging dir.
  build_aot_cache.py capture --stage build/aot -- \
      python -m vllm.entrypoints.openai.api_server ...

  # 2. Check the staged cache covers what production launches.
  build_aot_cache.py verify --stage build/aot

  # 3. Move it into the package tree so setup.py packages it.
  build_aot_cache.py install --stage build/aot

``capture`` sets ``SPARKINFER_COMPILE_CACHE_DIR`` and ``CUTE_DSL_ARCH`` for the
child and nothing else, because *every* other ``SPARKINFER_*`` / ``CUTE_*`` /
``CUTLASS_*`` variable present in the environment is part of the compile cache
key (``sparkinfer/_lib/compiler.py:_compile_environment_key``).  A capture run
whose environment differs from the serving environment in any such variable
produces objects the serving process cannot select.  ``capture`` therefore
records the environment it saw and ``verify`` reports it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from sparkinfer._lib.aot import AOT_CACHE_DIRNAME  # noqa: E402
from sparkinfer._lib.aot_matrix import check_coverage  # noqa: E402

_PACKAGE_CACHE = _REPO_ROOT / "sparkinfer" / AOT_CACHE_DIRNAME
_ENV_SNAPSHOT = "capture_env.json"


def _key_env(environ: dict[str, str]) -> dict[str, str]:
    """The environment variables that participate in the compile cache key."""
    return {
        name: value
        for name, value in sorted(environ.items())
        if name.startswith(("SPARKINFER_", "CUTE_", "CUTLASS_", "NVCC_"))
        or name in {"CC", "CXX", "CUDA_HOME", "CUDA_PATH", "CUDACXX"}
    }


def cmd_capture(args: argparse.Namespace) -> int:
    stage = Path(args.stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["SPARKINFER_COMPILE_CACHE_DIR"] = str(stage)
    env["SPARKINFER_COMPILE_DISK_CACHE"] = "1"
    if args.arch:
        env["CUTE_DSL_ARCH"] = args.arch
    # A capture run is *supposed* to compile; the AOT miss guard must not turn
    # that into an error.
    env["SPARKINFER_REQUIRE_AOT"] = "0"

    (stage / _ENV_SNAPSHOT).write_text(
        json.dumps(_key_env(env), indent=2, sort_keys=True) + "\n"
    )

    print(f"[aot] capturing into {stage}", flush=True)
    completed = subprocess.run(args.command, env=env)
    print(f"[aot] warmup exited {completed.returncode}", flush=True)
    return completed.returncode


def _manifests(stage: Path) -> list[dict]:
    out = []
    for path in sorted(stage.rglob("*.json")):
        if path.name == _ENV_SNAPSHOT:
            continue
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return out


def _observed_coverage(stage: Path) -> dict[str, int]:
    """kernel_id -> number of distinct compiled configurations."""
    by_kernel: dict[str, set[str]] = defaultdict(set)
    for manifest in _manifests(stage):
        kernel_id = manifest.get("kernel_id") or ""
        if not kernel_id:
            continue
        by_kernel[kernel_id].add(str(manifest.get("compile_spec_hash", "")))
    return {kernel: len(specs) for kernel, specs in by_kernel.items()}


def cmd_verify(args: argparse.Namespace) -> int:
    stage = Path(args.stage).resolve()
    objects = list(stage.rglob("*.o"))
    manifests = _manifests(stage)
    observed = _observed_coverage(stage)

    print(f"[aot] {stage}: {len(objects)} object(s), {len(manifests)} manifest(s)")
    for kernel, count in sorted(observed.items()):
        print(f"[aot]   {kernel}: {count} configuration(s)")

    snapshot = stage / _ENV_SNAPSHOT
    if snapshot.is_file():
        captured = json.loads(snapshot.read_text())
        print("[aot] capture environment (all of it is in the cache key):")
        for name, value in sorted(captured.items()):
            print(f"[aot]   {name}={value}")

    if not objects:
        print(
            "[aot] FAIL: no objects. The warmup never reached a cute.compile, or "
            "SPARKINFER_COMPILE_CACHE_DIR did not take effect.",
            file=sys.stderr,
        )
        return 1

    problems = check_coverage(observed)
    if problems:
        print("[aot] FAIL: shipped cache would not cover production:", file=sys.stderr)
        for problem in problems:
            print(f"[aot]   - {problem}", file=sys.stderr)
        return 1

    print("[aot] ok: coverage contract satisfied")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    stage = Path(args.stage).resolve()
    if not stage.is_dir():
        print(f"[aot] FAIL: {stage} does not exist", file=sys.stderr)
        return 1

    if _PACKAGE_CACHE.exists():
        shutil.rmtree(_PACKAGE_CACHE)
    _PACKAGE_CACHE.mkdir(parents=True)

    copied = 0
    for source in sorted(stage.rglob("*")):
        if source.is_dir() or source.name == _ENV_SNAPSHOT:
            continue
        if source.suffix not in {".o", ".json"}:
            continue  # .lock and .tmp files are never shipped
        destination = _PACKAGE_CACHE / source.relative_to(stage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    print(f"[aot] installed {copied} file(s) into {_PACKAGE_CACHE}")
    return 0 if copied else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    capture = sub.add_parser("capture", help="run a warmup with the cache redirected")
    capture.add_argument("--stage", required=True)
    capture.add_argument(
        "--arch",
        default="",
        help="CUTE_DSL_ARCH for the child (e.g. sm_121a). Empty = detect.",
    )
    capture.add_argument("command", nargs=argparse.REMAINDER)
    capture.set_defaults(func=cmd_capture)

    verify = sub.add_parser("verify", help="check the staged cache covers production")
    verify.add_argument("--stage", required=True)
    verify.set_defaults(func=cmd_verify)

    install = sub.add_parser("install", help="copy the staged cache into the package")
    install.add_argument("--stage", required=True)
    install.set_defaults(func=cmd_install)

    args = parser.parse_args()
    if args.cmd == "capture":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            parser.error("capture needs a warmup command after --")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
