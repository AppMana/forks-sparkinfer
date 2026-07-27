"""Build the PCIe comm CUDA extensions at install time.

Without this the extensions are compiled by torch.utils.cpp_extension.load()
on first call, which moves an nvcc run into the first inference request and
requires a compiler and a writable cache on the serving node.

Falls back to a pure-Python install when torch is unavailable at build time;
sparkinfer/comm/pcie/_ext.py then JIT-compiles on demand.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

_PCIE = Path("sparkinfer/comm/pcie")

# module name -> source, matching the names load_ext() imports.
_EXTENSIONS = {
    "sparkinfer_pcie_dma_ext": _PCIE / "pcie_dma.cu",
    "sparkinfer_pcie_oneshot_ext": _PCIE / "pcie_oneshot.cu",
    "sparkinfer_pcie_twoshot_ext": _PCIE / "pcie_twoshot.cu",
    "sparkinfer_pcie_dcp_a2a_ext": _PCIE / "pcie_dcp_a2a.cu",
}

# Set to "1" to skip the extension build (CPU-only or docs builds).
_SKIP = os.getenv("SPARKINFER_SKIP_EXT_BUILD", "0") == "1"


def _build_kwargs() -> dict:
    if _SKIP:
        return {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
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
        return {}
    return {
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": BuildExtension.with_options(use_ninja=True)},
    }


setup(**_build_kwargs())
