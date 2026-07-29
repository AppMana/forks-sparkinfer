#!/usr/bin/env python3
"""Stage prebuilt CuTe-DSL kernel objects into the package for the wheel.

This populates a cache; it does not establish a contract.  Anything it misses
compiles on first use exactly as sparkinfer always did, so a partial capture is
a partial speed-up and never a failure.

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

What is *not* device-free is reaching sparkinfer's ``compile()`` calls.  The
three attention families pass ``compile_args = runtime_args`` -- the compile
arguments are live ``from_dlpack`` tensors over real CUDA memory, at
``attention/_shared/mla/kernel.py:2449-2493``,
``attention/_shared/mla/merge.py:556-561`` and
``attention/_shared/mla/prefill_mg.py:3767-3786``.  Feeding those from
shape-only phantoms is possible -- ``merge.py`` already does exactly that for
its cache *key*, via ``_shape_only_scratch_tensor``
(``attention/compressed_mla/_scratch.py:235-242``) -- but it is a real change to
three hot launch paths, and since a missed configuration now costs one compile
rather than an outage, it is not worth that risk today.  This script therefore
captures the cache from a real warmup on target hardware.

(The two MoE families are already device-free: ``_get_micro_kernel``
(``moe/fused_moe/_impl.py:6586``) and ``_get_dynamic_kernel`` (``:7189``) take
only scalars and use fake pointers throughout.  See
``sparkinfer/_lib/aot_matrix.py`` for the follow-up that would exploit that.)

Usage
-----
  # 1. On a machine with the target GPU: run any warmup that exercises the
  #    production shapes -- normally the vLLM engine's own startup and CUDA
  #    graph capture -- with the compile cache redirected to a staging dir.
  build_aot_cache.py capture --stage build/aot --arch sm_121a -- \
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
from sparkinfer._lib.aot_matrix import TARGET_ARCHS, check_coverage  # noqa: E402

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
        # Setting this pins what the DSL code-generates for and, because the
        # cache key carries the *resolved* arch, it must be one of the names a
        # live device resolves to -- otherwise the objects key on a string
        # nothing at runtime will ever produce.
        if args.arch not in TARGET_ARCHS:
            print(
                f"[aot] FAIL: --arch {args.arch} is not one of {list(TARGET_ARCHS)}. "
                "sparkinfer only runs on compute capability 12.0 and 12.1 "
                "(sparkinfer/_lib/gating.py:29-32).",
                file=sys.stderr,
            )
            return 1
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


def cmd_precompile(args: argparse.Namespace) -> int:
    """Compile the configuration matrix with no GPU present.

    This is the primary way to populate the cache. It drives the real launch
    paths with fabricated argument descriptors (sparkinfer/_lib/aot_args.py),
    so the compile specs are computed by the same code production uses -- but
    nothing is allocated and nothing is executed.
    """
    import os

    stage = Path(args.stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)

    if args.arch not in TARGET_ARCHS:
        print(
            f"[aot] FAIL: --arch {args.arch} is not one of {list(TARGET_ARCHS)}.",
            file=sys.stderr,
        )
        return 1

    os.environ["CUTE_DSL_ARCH"] = args.arch
    os.environ["SPARKINFER_COMPILE_CACHE_DIR"] = str(stage)
    os.environ["SPARKINFER_COMPILE_DISK_CACHE"] = "1"
    os.environ["SPARKINFER_REQUIRE_AOT"] = "0"
    os.environ.setdefault("SPARKINFER_AOT_NUM_SM", str(args.sm_count))

    (stage / _ENV_SNAPSHOT).write_text(
        json.dumps(_key_env(dict(os.environ)), indent=2, sort_keys=True) + "\n"
    )

    import warnings

    from sparkinfer._lib.aot import AotCacheMissWarning
    from sparkinfer._lib.aot_precompile import Deployment, precompile

    # Every compile here is by definition a miss; the per-kernel note is the
    # thing this command exists to eliminate later, not to hear now.
    warnings.simplefilter("ignore", AotCacheMissWarning)

    print(f"[aot] precompiling for {args.arch} into {stage}", flush=True)
    results = precompile(Deployment(sm_count=args.sm_count))

    failed = [r for r in results if not r.ok]
    print(
        f"[aot] {len(results) - len(failed)}/{len(results)} configurations compiled",
        flush=True,
    )
    if failed:
        # Not fatal: a configuration that will not compile ahead of time simply
        # compiles on first use, exactly as before. It is reported so it can be
        # fixed, and so nobody believes coverage they do not have.
        print(
            f"[aot] {len(failed)} configuration(s) could not be compiled ahead "
            "of time and will JIT on first use:",
            file=sys.stderr,
        )
        for result in failed:
            print(f"[aot]   - {result.name}: {result.detail}", file=sys.stderr)
    return 0


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
        print(
            "[aot] FAIL: this capture is below the coverage floor -- a warmup "
            "that used to exercise these kernels no longer does:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"[aot]   - {problem}", file=sys.stderr)
        return 1

    print("[aot] ok: capture meets the coverage floor")
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
        choices=("", *TARGET_ARCHS),
        help="CUTE_DSL_ARCH for the child. Empty = detect from the local GPU.",
    )
    capture.add_argument("command", nargs=argparse.REMAINDER)
    capture.set_defaults(func=cmd_capture)

    verify = sub.add_parser("verify", help="check the staged cache covers production")
    verify.add_argument("--stage", required=True)
    verify.set_defaults(func=cmd_verify)

    pre = sub.add_parser(
        "precompile", help="compile the matrix with no GPU (the normal path)"
    )
    pre.add_argument("--stage", required=True)
    pre.add_argument("--arch", required=True, choices=TARGET_ARCHS)
    pre.add_argument(
        "--sm-count",
        type=int,
        default=48,
        help="SMs on the target. 48 = GB10. Enters the decode split policy.",
    )
    pre.set_defaults(func=cmd_precompile)

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
