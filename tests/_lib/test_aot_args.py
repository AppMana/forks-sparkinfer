"""The fabricated compile descriptors must be indistinguishable from the real ones.

This is the load-bearing test of the GPU-less AOT build.  sparkinfer selects a
cached object by its own ``KernelCompileSpec``, never by the cute signature, so
a fabricated descriptor that differs from what ``_to_cute`` builds would still
be selected at runtime -- and then handed arguments it was not compiled for.
Nothing downstream would catch that.  This does.

Needs a CUDA device only to build the *real* side of the comparison; the
architecture is irrelevant, since MLIR types carry no arch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
cutlass = pytest.importorskip("cutlass")

from sparkinfer._lib.aot_args import (  # noqa: E402
    FakeCudaTensor,
    TensorSpec,
    fabricate_cute_tensor,
    to_cute_arg,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the real side of the comparison needs from_dlpack over device memory",
)


# Every distinct (rank, dtype, alignment, dynamic-layout) combination the three
# MLA launch paths build. Names are the kernel argument names at
# attention/_shared/mla/kernel.py:2449-2493, merge.py:556-561 and
# prefill_mg.py:3767-3786.
_DESCRIPTORS = [
    ("q_all", (12, 32, 576), torch.bfloat16, cutlass.BFloat16, 16, True),
    ("kv_flat", (1 << 20,), torch.uint8, cutlass.Uint8, 16, False),
    ("swa_indices", (12, 512), torch.int32, cutlass.Int32, 4, True),
    ("mid_out", (12, 32, 2, 512), torch.bfloat16, cutlass.BFloat16, 16, True),
    ("mid_lse", (12, 32, 2), torch.float32, cutlass.Float32, 4, True),
    ("swa_len", (12,), torch.int32, cutlass.Int32, 4, True),
    ("merge_out", (12, 32, 512), torch.bfloat16, cutlass.BFloat16, 16, True),
    ("merge_lse", (12, 32), torch.float32, cutlass.Float32, 4, True),
    ("sink", (32,), torch.float32, cutlass.Float32, 4, True),
]


def _mlir_type(tensor_like) -> str:
    from cutlass._mlir import ir

    with ir.Context(), ir.Location.unknown():
        return str(tensor_like.mlir_type)


@pytest.mark.parametrize(
    "name,shape,torch_dtype,cute_dtype,align,dynamic", _DESCRIPTORS
)
def test_fabricated_descriptor_matches_from_dlpack(
    name: str, shape, torch_dtype, cute_dtype, align: int, dynamic: bool
) -> None:
    real_tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")
    real = to_cute_arg(real_tensor, cute_dtype, align=align, dynamic_layout=dynamic)

    fake_tensor = FakeCudaTensor(shape, torch_dtype, stride=tuple(real_tensor.stride()))
    fake = to_cute_arg(fake_tensor, cute_dtype, align=align, dynamic_layout=dynamic)

    assert _mlir_type(fake) == _mlir_type(real), (
        f"{name}: fabricated descriptor differs from from_dlpack's. A cached "
        f"object built from this would be selected at runtime and handed "
        f"arguments it was not compiled for."
    )


def test_symint_width_rule_is_the_one_that_matches() -> None:
    """Pin the rule, because it is measured and not derivable.

    Dynamic extents are i32 and render ``?``; dynamic strides are i64 and
    render ``?{i64}``. Widening the shape to i64 produces a type that looks
    plausible and is wrong.
    """
    from cutlass.cute.runtime import make_fake_tensor
    from cutlass.cute.typing import SymInt

    shape, stride = (12, 32, 576), (18432, 576, 1)
    real = to_cute_arg(
        torch.empty(shape, dtype=torch.bfloat16, device="cuda"),
        cutlass.BFloat16,
        align=16,
        dynamic_layout=True,
    )
    wrong = make_fake_tensor(
        cutlass.BFloat16,
        tuple(SymInt(width=64) for _ in shape),
        (SymInt(width=64), SymInt(width=64), 1),
        assumed_align=16,
    )
    assert _mlir_type(wrong) != _mlir_type(real)
    assert "?{i64},?{i64},?{i64}" in _mlir_type(wrong)

    right = fabricate_cute_tensor(
        TensorSpec(shape=shape, stride=stride, align=16, dynamic_layout=True),
        cutlass.BFloat16,
    )
    assert _mlir_type(right) == _mlir_type(real)


def test_static_layout_keeps_its_extents() -> None:
    """A non-dynamic descriptor bakes the extent in; the fake must too."""
    real = to_cute_arg(
        torch.empty(1 << 20, dtype=torch.uint8, device="cuda"),
        cutlass.Uint8,
        align=16,
        dynamic_layout=False,
    )
    fake = to_cute_arg(
        FakeCudaTensor((1 << 20,), torch.uint8),
        cutlass.Uint8,
        align=16,
        dynamic_layout=False,
    )
    assert _mlir_type(fake) == _mlir_type(real)
    assert "1048576" in _mlir_type(fake)


def test_fake_tensor_refuses_to_hand_out_a_pointer() -> None:
    with pytest.raises(RuntimeError, match="no storage"):
        FakeCudaTensor((4,), torch.float32).data_ptr()


def test_fake_tensor_reports_cuda_so_the_compile_key_matches() -> None:
    """A meta tensor would key differently and never be selected."""
    fake = FakeCudaTensor((4, 8), torch.float32)
    assert fake.device.type == "cuda"
    assert fake.stride() == (8, 1)
    assert fake.is_contiguous()
