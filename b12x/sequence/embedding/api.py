"""Exact unquantized or compact Q8_0/Q5_K embedding lookup without output or replay allocation."""
import torch

from ..._lib.gating import default_is_supported
from . import META
from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from ._preparation import plan, query_from_call
from ._tuning import EmbeddingQuery
from ._q5 import Q5KEmbeddingWeight


def _check_weight(weight):
    if not weight.is_cuda or weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("embedding weight must be CUDA BF16 or FP32")
    if weight.ndim != 2 or weight.shape[1] <= 0:
        raise ValueError("embedding weight must be [table_rows, positive width]")
    if weight.stride(1) != 1 or weight.stride(0) < weight.shape[1]:
        raise ValueError("embedding weight must have contiguous nonoverlapping rows")


def _check_tensors(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                   num_rows: torch.Tensor | None = None) -> None:
    _check_weight(weight)
    if ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous():
        raise TypeError("embedding IDs must be contiguous Int32 or Int64")
    if ids.device != weight.device or out.device != weight.device:
        raise ValueError("embedding tensors must share the CUDA device")
    if out.shape != (*ids.shape, weight.shape[1]) or out.dtype != weight.dtype:
        raise ValueError("embedding output must have ids.shape + (width,) and weight dtype")
    if not out.is_contiguous() or ids.numel() >= 2**31:
        raise ValueError("embedding requires contiguous output and an Int32 launch count")
    if torch._C._overlaps(out, weight) or torch._C._overlaps(out, ids):
        raise ValueError("embedding output must not alias inputs")
    if num_rows is not None:
        if (num_rows.device != weight.device or num_rows.dtype != torch.int32
                or num_rows.numel() != 1 or not num_rows.is_contiguous()):
            raise ValueError("num_rows must be a CUDA Int32 scalar on the weight device")
        if ids.numel() == 0:
            raise ValueError("device-count lookup requires positive output capacity")
        if torch._C._overlaps(out, num_rows):
            raise ValueError("embedding output must not alias num_rows")


@torch.library.custom_op("b12x::embedding_out", mutates_args=("out",))
def _embedding_out(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                   plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    require_prepared(plan_from_handle(plan_handle), "sequence.embedding", weight.device).run(
        weight, ids, out=out, num_rows=num_rows,
    )


@_embedding_out.register_fake
def _embedding_out_fake(weight: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                        plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    return None


@torch.library.custom_op("b12x::embedding_q8_out", mutates_args=("out",))
def _embedding_q8_out(values: torch.Tensor, metadata: torch.Tensor, ids: torch.Tensor,
                      out: torch.Tensor, plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    from b12x.gemm.blockscaled._q8_0 import Q8_0LinearWeight
    weight = Q8_0LinearWeight(values, metadata, values.shape[1], values.shape[0])
    require_prepared(plan_from_handle(plan_handle), "sequence.embedding", values.device).run(
        weight, ids, out=out, num_rows=num_rows)


@_embedding_q8_out.register_fake
def _embedding_q8_out_fake(values: torch.Tensor, metadata: torch.Tensor, ids: torch.Tensor,
                           out: torch.Tensor, plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    return None


@torch.library.custom_op("b12x::embedding_q5_out", mutates_args=("out",))
def _embedding_q5_out(values: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                      plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    require_prepared(plan_from_handle(plan_handle), "sequence.embedding", values.device).run(
        Q5KEmbeddingWeight(values), ids, out=out, num_rows=num_rows)


@_embedding_q5_out.register_fake
def _embedding_q5_out_fake(values: torch.Tensor, ids: torch.Tensor, out: torch.Tensor,
                           plan_handle: int, num_rows: torch.Tensor | None = None) -> None:
    return None


def run(weight, ids, *, out, plan: Plan, num_rows=None):
    """Gather rows (Q8_0 decoded to BF16) into caller storage through the prepared embedding kernel.

    IDs and the optional device live count remain dynamic during graph replay.
    Only live rows are written. Invalid live IDs or counts trap on the device.
    """
    from b12x.gemm.blockscaled._q8_0 import Q8_0LinearWeight
    if isinstance(weight, Q5KEmbeddingWeight):
        _embedding_q5_out(weight.values, ids, out, plan.handle, num_rows)
    elif isinstance(weight, Q8_0LinearWeight):
        _embedding_q8_out(weight.values, weight.metadata, ids, out, plan.handle, num_rows)
    else:
        _embedding_out(weight, ids, out, plan.handle, num_rows)
    return out


def is_supported(device=None):
    return default_is_supported(device, requires=META.requires)


def clear_caches():
    from ._kernel import compile_embedding
    compile_embedding.cache_clear()
    from ._q8 import compile_q8_embedding
    compile_q8_embedding.cache_clear()
    from ._q5 import compile_q5_embedding
    compile_q5_embedding.cache_clear()
