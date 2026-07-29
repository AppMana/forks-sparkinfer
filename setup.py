"""Build the PCIe comm CUDA extensions at install time and record the binding.

Without this the extensions are compiled by torch.utils.cpp_extension.load()
on first call, which moves an nvcc run into the first inference request and
requires a compiler and a writable cache on the serving node.

Two things are emitted besides the extensions:

* ``sparkinfer/_build_info.json`` -- the torch version, torch's CUDA version
  and the arch lists this wheel was built against.  The wheel filename records
  none of them (there is no torch or CUDA field in a PEP 425 tag), so the
  binding is recorded inside the wheel and enforced at load time by
  ``sparkinfer._lib.build_info``.  Publishing a wheel built against a different
  torch under the same version is still *possible*; it is no longer *silent*.
* ``sparkinfer/_aot_cache/`` -- prebuilt CuTe-DSL kernel objects, staged by
  ``scripts/build_aot_cache.py`` before this runs.  Present or absent it is
  packaged verbatim; the CI workflow is what asserts it is non-empty.

Falls back to a pure-Python install when torch is unavailable at build time.
torch is in build-system.requires, so an isolated PEP 517 build reaches the
real path and that fallback is now only for a deliberately torch-less
environment (SPARKINFER_SKIP_EXT_BUILD, or a docs build). It stays silent by
design -- setup.py cannot know whether the caller wanted extensions -- so the
loudness lives one level up, in the in-wheel assertions in
.github/workflows/wheels.yaml.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from setuptools import setup

_PACKAGE = Path("sparkinfer")
_PCIE = _PACKAGE / "comm" / "pcie"

# module name -> source, matching the names load_ext() imports.
_EXTENSIONS = {
    "sparkinfer_pcie_dma_ext": _PCIE / "pcie_dma.cu",
    "sparkinfer_pcie_oneshot_ext": _PCIE / "pcie_oneshot.cu",
    "sparkinfer_pcie_twoshot_ext": _PCIE / "pcie_twoshot.cu",
    "sparkinfer_pcie_dcp_a2a_ext": _PCIE / "pcie_dcp_a2a.cu",
}

# Set to "1" to skip the extension build (CPU-only or docs builds).
_SKIP = os.getenv("SPARKINFER_SKIP_EXT_BUILD", "0") == "1"


def _write_build_info(extensions_built: bool) -> None:
    """Record what this wheel is bound to, next to the code that checks it."""
    info: dict[str, object] = {
        "extensions_built": extensions_built,
        "torch_cuda_arch_list": os.getenv("TORCH_CUDA_ARCH_LIST", ""),
        "cute_dsl_arch": os.getenv("CUTE_DSL_ARCH", ""),
    }
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["torch_cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
    except ImportError:
        info["torch_version"] = ""
        info["torch_cuda_version"] = ""
    try:
        import importlib.metadata as _metadata

        info["cutlass_dsl_version"] = _metadata.version("nvidia-cutlass-dsl")
    except Exception:
        info["cutlass_dsl_version"] = ""

    (_PACKAGE / "_build_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n"
    )


def _build_kwargs() -> dict:
    if _SKIP:
        _write_build_info(False)
        return {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        _write_build_info(False)
        return {}

    # TORCH_CUDA_ARCH_LIST governs the SASS; the caller sets it (the image
    # passes 8.6 and 12.1a).
    ext_modules = [
        CUDAExtension(
            name=name,
            sources=[str(src)],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
            libraries=["cuda"],
        )
        for name, src in _EXTENSIONS.items()
        if src.exists()
    ]
    if not ext_modules:
        _write_build_info(False)
        return {}
    _write_build_info(True)
    return {
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": BuildExtension.with_options(use_ninja=True)},
    }


setup(**_build_kwargs())
