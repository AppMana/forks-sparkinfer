"""Fabricated compile-time argument descriptors, so AOT needs no GPU.

The problem
-----------
``cute.compile`` only ever reads a tensor argument's *type*: dtype, address
space, alignment and layout.  It never touches the memory.  But the three MLA
launch paths pass the same tuple as both ``compile_args`` and ``runtime_args``,
so the compile-time signature was pinned to live ``from_dlpack`` tensors over
real CUDA allocations -- which meant building the AOT cache required a GPU of
the target architecture even though CUTLASS itself can code-generate for one it
cannot see.

``moe.fused_moe`` already avoided this: it compiles against ``make_ptr`` fake
pointers and ``make_fake_compact_tensor`` descriptors
(``moe/fused_moe/_impl.py:6676-6695``, ``:7499-7551``).  This module generalises
that pattern to the layouts the MLA kernels use, which the MoE helpers do not
cover because those are all compact and these are not.

The equivalence, measured
-------------------------
``_to_cute`` builds ``from_dlpack(t).mark_layout_dynamic(leading_dim=k)``.
Reproducing that exactly matters more than anything else here: sparkinfer
selects a cached object by its own ``KernelCompileSpec``, not by the cute
signature, so a fabricated descriptor that differs from the real one would be
silently selected at runtime and handed arguments it was not compiled for.

Compared as MLIR types on an sm_86 device, for every descriptor shape the three
MLA sites use::

    from_dlpack(q_all).mark_layout_dynamic(leading_dim=2)
      -> !cute.memref<bf16, gmem, align<16>, "(?,?,?):(?{i64},?{i64},1)">
    fabricate(BFloat16, (12,32,576), (18432,576,1), align=16, dynamic=True)
      -> !cute.memref<bf16, gmem, align<16>, "(?,?,?):(?{i64},?{i64},1)">

The rule that makes them identical, and it is not guessable:

* dynamic **shape** extents are ``SymInt(width=32)``  -- these render ``?``
* dynamic **stride** entries are ``SymInt(width=64)`` -- these render ``?{i64}``
* the stride at ``leading_dim`` stays the literal ``1``

``SymInt(width=64)`` in the shape renders ``?{i64}`` and does *not* match.
``tests/_lib/test_aot_args.py`` pins all of this against real tensors and is
skipped when no CUDA device is present.

Note that ``__cache_key__`` is useless as an oracle here: ``_FakeTensor``'s
includes ``id()`` of its ``SymInt`` objects, so two structurally identical fake
descriptors compare unequal.  The MLIR type is the only sound comparison.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from threading import local
from typing import Any

_STATE = local()


def compile_only_mode() -> bool:
    """True while an AOT build is driving the launch paths.

    Set only by ``scripts/build_aot_cache.py``.  In serving this is always
    False and every path below behaves exactly as it did before this module
    existed.
    """
    return bool(getattr(_STATE, "compile_only", False))


@contextmanager
def compile_only():
    """Compile the kernel and skip the launch.

    Used by the AOT builder: the launch paths run normally right up to
    ``cute.compile``, then stop rather than trying to execute against
    descriptors that have no memory behind them.
    """
    previous = getattr(_STATE, "compile_only", False)
    _STATE.compile_only = True
    try:
        yield
    finally:
        _STATE.compile_only = previous


class FakeCudaTensor:
    """Enough of a CUDA ``torch.Tensor`` to describe one, and nothing more.

    The MLA launch paths use their tensor arguments for exactly two things:
    metadata (``.shape``, ``.stride()``, ``.dtype``, ``.device``, read by
    ``tensor_key`` to build the compile spec) and conversion to a cute
    descriptor (``_to_cute`` / ``_to_kernel_tensor``).  This satisfies the
    first; ``to_cute_arg`` below intercepts the second.

    It reports ``device='cuda:0'`` deliberately.  ``tensor_compile_fact``
    records the device type in the compile key
    (``sparkinfer/_lib/compiler.py:tensor_compile_fact``), so a descriptor
    claiming ``meta`` -- which is why a torch meta tensor cannot be used here --
    would key differently from the serving process and never be selected.

    Any attempt to read data raises: there is none, and a silent zero would be
    far worse than a traceback.
    """

    __slots__ = ("_shape", "_stride", "dtype", "device")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        *,
        stride: tuple[int, ...] | None = None,
        device: str = "cuda:0",
    ) -> None:
        import torch

        self._shape = torch.Size(tuple(int(d) for d in shape))
        self._stride = (
            tuple(int(s) for s in stride)
            if stride is not None
            else _contiguous_stride(tuple(int(d) for d in shape))
        )
        self.dtype = dtype
        self.device = torch.device(device)

    @property
    def shape(self) -> Any:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def size(self, dim: int | None = None) -> Any:
        return self._shape if dim is None else self._shape[dim]

    def stride(self, dim: int | None = None) -> Any:
        return self._stride if dim is None else self._stride[dim]

    def numel(self) -> int:
        total = 1
        for extent in self._shape:
            total *= int(extent)
        return total

    def is_contiguous(self) -> bool:
        return self._stride == _contiguous_stride(tuple(self._shape))

    def element_size(self) -> int:
        import torch

        return torch.empty(0, dtype=self.dtype).element_size()

    def data_ptr(self) -> int:
        raise RuntimeError(
            "FakeCudaTensor has no storage; an AOT compile path tried to read a "
            "device pointer. Fabricate the compile descriptor instead of "
            "dereferencing the tensor."
        )

    def __repr__(self) -> str:
        return (
            f"FakeCudaTensor(shape={tuple(self._shape)}, stride={self._stride}, "
            f"dtype={self.dtype}, device={self.device})"
        )


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        stride[idx] = stride[idx + 1] * int(shape[idx + 1])
    return tuple(stride)


@dataclass(frozen=True)
class TensorSpec:
    """Everything ``cute.compile`` can observe about a tensor argument."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    align: int
    dynamic_layout: bool


def fabricate_cute_tensor(spec: TensorSpec, dtype: Any) -> Any:
    """Build the descriptor ``_to_cute`` would have built, with no memory.

    See the module docstring for the width rule; it is measured, not derived.
    """
    from cutlass.cute.runtime import make_fake_tensor
    from cutlass.cute.typing import SymInt

    shape: tuple[Any, ...] = spec.shape
    stride: tuple[Any, ...] = spec.stride

    if spec.dynamic_layout and spec.shape:
        leading_dim = next(
            (idx for idx, value in enumerate(spec.stride) if value == 1), None
        )
        if leading_dim is not None:
            shape = tuple(SymInt(width=32) for _ in spec.shape)
            stride = tuple(
                1 if idx == leading_dim else SymInt(width=64)
                for idx in range(len(spec.stride))
            )

    return make_fake_tensor(dtype, shape, stride, assumed_align=spec.align)


def spec_of(tensor: Any, *, align: int, dynamic_layout: bool) -> TensorSpec:
    return TensorSpec(
        shape=tuple(int(d) for d in tensor.shape),
        stride=tuple(int(s) for s in tensor.stride()),
        align=int(align),
        dynamic_layout=bool(dynamic_layout),
    )


def to_cute_arg(
    tensor: Any,
    dtype: Any,
    *,
    align: int,
    dynamic_layout: bool,
    min_ndim: int = 1,
) -> Any:
    """The single seam between a real launch and an AOT compile.

    A real ``torch.Tensor`` takes the original ``from_dlpack`` path, byte for
    byte unchanged.  A :class:`FakeCudaTensor` -- which only the AOT builder
    ever constructs -- is fabricated instead.  Serving never reaches the second
    branch.

    ``min_ndim`` exists because the three call sites disagree: ``kernel.py``
    and ``prefill_mg.py`` mark any rank >= 1 dynamic, ``merge.py`` only rank
    >= 2.  That difference is preserved rather than normalised -- it changes
    the emitted layout, so unifying it would silently recompile every merge
    kernel.
    """
    dynamic = bool(dynamic_layout) and tensor.ndim >= min_ndim

    if isinstance(tensor, FakeCudaTensor):
        return fabricate_cute_tensor(
            spec_of(tensor, align=align, dynamic_layout=dynamic), dtype
        )

    from cutlass.cute.runtime import from_dlpack

    cute_tensor = from_dlpack(tensor, assumed_align=align)
    cute_tensor.element_type = dtype
    if dynamic:
        leading_dim = next(
            (idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None
        )
        if leading_dim is not None:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def compile_stream() -> Any:
    """The stream handle to compile against.

    ``cute.compile`` records only that the host wrapper takes a stream, so a
    fake one is signature-equivalent.  Outside an AOT build this returns the
    live stream exactly as before -- a launch still needs the real handle.
    """
    if compile_only_mode():
        from cutlass.cute.runtime import make_fake_stream

        return make_fake_stream()

    from sparkinfer._lib.utils import current_cuda_stream

    return current_cuda_stream()


def aot_num_sm() -> int | None:
    """SM count for an AOT build, or None to query the device.

    ``sm_count`` reaches a compile key through ``_wave_balanced_num_splits``
    and through the MoE ``m1_fc2_onepass`` path, so a GPU-less build has to be
    told what it is targeting. GB10 has 48.
    """
    raw = os.environ.get("SPARKINFER_AOT_NUM_SM", "")
    if not raw:
        return None
    return int(raw)
