"""Extension loading for the PCIe comm kernels."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Set to "1" to skip the prebuilt module and force a JIT rebuild, for editing a
# .cu without reinstalling the package.
_FORCE_JIT_ENV = "SPARKINFER_FORCE_JIT_EXT"


def load_ext(
    name: str,
    source: Path | str,
    *,
    extra_cuda_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
    verbose: bool = False,
) -> Any:
    """Return the compiled extension ``name``.

    Imports the module built by ``setup.py``. Falls back to
    ``torch.utils.cpp_extension.load()`` for source checkouts where the
    extensions were not built. ``name`` is both the AOT module name and the
    JIT build name.
    """
    if os.getenv(_FORCE_JIT_ENV, "0") != "1":
        try:
            return importlib.import_module(name)
        except ImportError:
            pass

    from torch.utils.cpp_extension import load

    logger.warning(
        "%s was not built ahead of time; compiling now. Install sparkinfer "
        "from a wheel or with `pip install .` to build it at install time.",
        name,
    )
    return load(
        name=name,
        sources=[str(source)],
        extra_cuda_cflags=extra_cuda_cflags or ["-O2"],
        extra_ldflags=extra_ldflags or [],
        verbose=verbose,
    )
