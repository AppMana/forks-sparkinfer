"""What the installed wheel was actually built against, and enforcing it.

A sparkinfer wheel carries C++ extensions linked against a specific torch C++
ABI and a specific CUDA major version, and CuTe-DSL objects code-generated for
a specific GPU architecture by a specific CUTLASS.  None of that appears in the
wheel filename -- ``sparkinfer-1.0.1-cp312-cp312-manylinux_2_28_aarch64.whl``
says ``cp312`` and ``aarch64`` and nothing else.  Two wheels built against
different torch versions are therefore indistinguishable to pip and to a human
reading a release page.

The filename cannot be fixed without leaving PEP 425 (there is no torch or CUDA
field in a wheel tag, and local version segments are rejected by most indexes).
So the binding is recorded *inside* the wheel instead, and checked at the point
where a mismatch would otherwise turn into an undefined-symbol ImportError that
``load_ext`` used to swallow into a silent nvcc rebuild.

Written by ``setup.py`` at build time; absent in a source checkout, which is
not an error -- it just means nothing was built AOT and there is nothing to
verify.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

BUILD_INFO_FILENAME = "_build_info.json"

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def build_info_path() -> Path:
    return _PACKAGE_ROOT / BUILD_INFO_FILENAME


@lru_cache(maxsize=1)
def build_info() -> dict[str, Any] | None:
    """Build metadata recorded by setup.py, or None for a source checkout."""
    path = build_info_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class BuildMismatch(ImportError):
    """The installed wheel was built against a different torch or CUDA."""


def _torch_runtime() -> tuple[str, str]:
    import torch

    return str(torch.__version__), str(getattr(torch.version, "cuda", "") or "")


def _major(version: str) -> str:
    return version.split(".", 1)[0] if version else ""


def check_runtime_compatibility(*, strict: bool = True) -> list[str]:
    """Return the list of build/runtime mismatches; raise on any when strict.

    Two things are compared, and only two, because only these can make a
    prebuilt artifact wrong rather than merely suboptimal:

    * **torch C++ ABI.** The PCIe extensions are ``CUDAExtension``s linked
      against ``libtorch``; torch does not promise a stable C++ ABI across
      minor releases, so the built version is compared exactly.
    * **CUDA major.** ``libcudart`` is a different soname across majors, and
      sm_121 does not exist as an nvcc target before CUDA 12.9, so a wheel
      built under CUDA 13 cannot be reused under CUDA 12.  Minor differences
      are compatible and are not compared.
    """
    info = build_info()
    if info is None:
        return []

    problems: list[str] = []
    try:
        torch_version, torch_cuda = _torch_runtime()
    except ImportError:
        return []

    built_torch = str(info.get("torch_version", ""))
    if built_torch and built_torch != torch_version:
        problems.append(
            f"torch: wheel was built against {built_torch}, running {torch_version}"
        )

    built_cuda = str(info.get("torch_cuda_version", ""))
    if built_cuda and _major(built_cuda) != _major(torch_cuda):
        problems.append(
            f"CUDA major: wheel was built against CUDA {built_cuda}, "
            f"torch reports CUDA {torch_cuda or '<none>'}"
        )

    if problems and strict:
        raise BuildMismatch(
            "sparkinfer's prebuilt binaries do not match this environment: "
            + "; ".join(problems)
            + ". The wheel filename does not encode either, so pip cannot have "
            "caught this. Install the wheel built for this torch/CUDA, or "
            "reinstall from source with `pip install --no-build-isolation .`."
        )
    return problems


def describe() -> dict[str, Any]:
    info = build_info() or {}
    return {
        "build_info_present": build_info() is not None,
        **{f"built_{k}": v for k, v in info.items()},
    }
